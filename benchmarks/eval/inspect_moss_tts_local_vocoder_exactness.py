# SPDX-License-Identifier: Apache-2.0
"""Trace exactness between upstream and owned MOSS-TTS Local vocoder decoders.

This is a debugging utility for the SGLang-backed MOSS vocoder work. It does
not change the serving path. It loads the real MOSS-TTS Local processor,
decodes deterministic codec-token probes through:

1. the upstream processor path, and
2. the repo-owned decoder path installed with ``use_moss_tts_local_vocoder_decoder``.

It captures comparable decoder-module boundaries and reports the first tensor
that diverges. The output is intended to answer one question before further
optimization: does the owned decoder preserve upstream vocoder semantics before
we route any supported attention call through SGLang?

Example:

    python -m benchmarks.eval.inspect_moss_tts_local_vocoder_exactness \
        --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
        --device cuda:0 \
        --output-dir /data/moss_vocoder_exactness \
        --probe 1x25 --probe 1x100 --probe 1x300 \
        --dump-first-mismatch
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

import sglang_omni.models.moss_tts_local.vocoder_decoder as vocoder_decoder_module
from sglang_omni.models.moss_tts_local.stages import _load_moss_tts_local_processor
from sglang_omni.models.moss_tts_local.vocoder_decoder import (
    build_moss_tts_local_vocoder_decoder,
    use_moss_tts_local_vocoder_decoder,
)

logger = logging.getLogger(__name__)
_SUPPRESS_SDPA_CAPTURE = 0


@dataclass
class _CapturedTensor:
    key: str
    shape: list[int]
    dtype: str
    device: str
    numel: int
    stored: bool
    tensor: torch.Tensor | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "numel": self.numel,
            "stored": self.stored,
        }


class _TraceCapture:
    """Forward-hook recorder with deterministic key names.

    Hooks are intentionally read-only: tensors are detached and copied to CPU
    after module execution. Large tensors can be summarized without storing the
    payload, which keeps long probes usable while still showing where trace
    coverage stopped being comparable.
    """

    def __init__(self, *, max_elements_per_tensor: int) -> None:
        self.max_elements_per_tensor = int(max_elements_per_tensor)
        self.records: "OrderedDict[str, _CapturedTensor]" = OrderedDict()
        self._handles: list[Any] = []
        self._call_counts: dict[int, int] = defaultdict(int)
        self._active_call: dict[int, list[int]] = defaultdict(list)

    def install(self, modules: list[tuple[str, nn.Module]]) -> None:
        seen: set[int] = set()
        for name, module in modules:
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            self._handles.append(
                module.register_forward_pre_hook(
                    self._make_pre_hook(name), with_kwargs=True
                )
            )
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_pre_hook(self, name: str):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            module_id = id(module)
            call_index = self._call_counts[module_id]
            self._call_counts[module_id] += 1
            self._active_call[module_id].append(call_index)
            prefix = f"{name}#{call_index:03d}.pre"
            self._capture_value(f"{prefix}.arg", args)
            for key in sorted(kwargs):
                self._capture_value(f"{prefix}.kw.{key}", kwargs[key])

        return hook

    def _make_hook(self, name: str):
        def hook(module: nn.Module, _: tuple[Any, ...], output: Any) -> None:
            module_id = id(module)
            stack = self._active_call[module_id]
            call_index = stack.pop() if stack else self._call_counts[module_id]
            self._capture_value(f"{name}#{call_index:03d}.post.out", output)

        return hook

    def _capture_value(self, prefix: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self._capture_tensor(prefix, value)
            return
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self._capture_value(f"{prefix}{index}", item)
            return
        if isinstance(value, dict):
            for key in sorted(value):
                self._capture_value(f"{prefix}.{key}", value[key])

    def _capture_tensor(self, key: str, tensor: torch.Tensor) -> None:
        detached = tensor.detach()
        numel = int(detached.numel())
        stored = numel <= self.max_elements_per_tensor
        cpu_tensor = detached.to("cpu") if stored else None
        self.records[key] = _CapturedTensor(
            key=key,
            shape=list(detached.shape),
            dtype=str(detached.dtype),
            device=str(detached.device),
            numel=numel,
            stored=stored,
            tensor=cpu_tensor,
        )


class _SdpaCapture:
    """Record PyTorch SDPA inputs and outputs without changing the call.

    Module hooks identify the first high-level boundary that diverges. This
    capture narrows attention mismatches to the actual SDPA contract: q, k, v,
    optional mask/bias tensor, and output. It patches the functional module
    attribute used by remote code that imports ``torch.nn.functional as F``.
    """

    def __init__(self, *, max_elements_per_tensor: int) -> None:
        self._capture = _TraceCapture(max_elements_per_tensor=max_elements_per_tensor)
        self._original = None
        self._call_index = 0

    @property
    def records(self) -> "OrderedDict[str, _CapturedTensor]":
        return self._capture.records

    def __enter__(self) -> "_SdpaCapture":
        self._original = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *_: object) -> None:
        if self._original is not None:
            F.scaled_dot_product_attention = self._original  # type: ignore[method-assign]
            self._original = None

    def _wrapped(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if _SUPPRESS_SDPA_CAPTURE:
            if self._original is None:
                raise RuntimeError("SDPA capture is not installed")
            return self._original(*args, **kwargs)
        call_index = self._call_index
        self._call_index += 1
        prefix = f"sdpa#{call_index:03d}"
        for index, name in enumerate(("query", "key", "value", "attn_mask")):
            if index < len(args):
                value = args[index]
            else:
                value = kwargs.get(name)
            if value is not None:
                self._capture._capture_value(f"{prefix}.pre.{name}", value)
        for name in ("dropout_p", "is_causal", "scale", "enable_gqa"):
            value = kwargs.get(name)
            if isinstance(value, torch.Tensor):
                self._capture._capture_value(f"{prefix}.pre.kw.{name}", value)
        if self._original is None:
            raise RuntimeError("SDPA capture is not installed")
        output = self._original(*args, **kwargs)
        self._capture._capture_value(f"{prefix}.post.out", output)
        return output


@contextmanager
def _suppress_sdpa_capture():
    global _SUPPRESS_SDPA_CAPTURE
    _SUPPRESS_SDPA_CAPTURE += 1
    try:
        yield
    finally:
        _SUPPRESS_SDPA_CAPTURE -= 1


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 32:
            return value.detach().to("cpu").reshape(-1).tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _sdpa_reference_from_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    max_seqlen_q: int,
    causal: bool,
    window_size: tuple[int, int],
    source_context: int | None,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    cu_q = [int(item) for item in cu_seqlens_q.detach().to("cpu").tolist()]
    cu_k = [int(item) for item in cu_seqlens_k.detach().to("cpu").tolist()]
    if len(cu_q) != len(cu_k):
        raise ValueError(
            f"cu_seqlens_q and cu_seqlens_k must have equal length, got "
            f"{len(cu_q)} and {len(cu_k)}"
        )

    for index in range(len(cu_q) - 1):
        q_start, q_end = cu_q[index], cu_q[index + 1]
        k_start, k_end = cu_k[index], cu_k[index + 1]
        q_i = q[q_start:q_end]
        k_i = k[k_start:k_end]
        v_i = v[k_start:k_end]
        query_len = q_i.shape[0]
        key_len = k_i.shape[0]
        q_sdpa = q_i.transpose(0, 1).unsqueeze(0)
        k_sdpa = k_i.transpose(0, 1).unsqueeze(0)
        v_sdpa = v_i.transpose(0, 1).unsqueeze(0)

        attn_mask = None
        if causal or source_context is not None or window_size != (-1, -1):
            q_positions = torch.arange(query_len, device=q.device, dtype=torch.long)
            k_positions = torch.arange(key_len, device=q.device, dtype=torch.long)
            delta = (key_len - query_len + q_positions).view(
                query_len, 1
            ) - k_positions.view(1, key_len)
            mask = torch.ones((query_len, key_len), device=q.device, dtype=torch.bool)
            if causal:
                mask = mask & (delta >= 0)
            if source_context is not None:
                mask = mask & (delta < int(source_context))
            elif window_size[0] >= 0:
                mask = mask & (delta <= int(window_size[0]))
            if source_context is None and window_size[1] >= 0:
                mask = mask & (delta >= -int(window_size[1]))
            attn_mask = mask.view(1, 1, query_len, key_len)

        with _suppress_sdpa_capture():
            out_i = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask,
                dropout_p=0.0,
            )
        outputs.append(out_i.squeeze(0).transpose(0, 1).contiguous())

    if not outputs:
        return q.new_empty((0, q.shape[1], q.shape[2]))
    total_q = cu_q[-1]
    if total_q != sum(item.shape[0] for item in outputs):
        raise RuntimeError(
            f"reconstructed SDPA produced unexpected token count for max_q="
            f"{max_seqlen_q}"
        )
    return torch.cat(outputs, dim=0)


class _FlashAttentionOracleCapture:
    """Compare owned SGLang varlen attention output to an SDPA oracle.

    This wraps the repo-owned MOSS attention adapter, records the exact packed
    tensors passed to SGLang, runs a segment-wise PyTorch SDPA reference over
    the same valid packed K/V rows, and leaves the production output unchanged.
    """

    def __init__(self, *, max_calls: int) -> None:
        self.max_calls = int(max_calls)
        self.calls: list[dict[str, Any]] = []
        self._original = None
        self._call_index = 0

    def __enter__(self) -> "_FlashAttentionOracleCapture":
        attention_cls = vocoder_decoder_module.MossTTSLocalAttention
        self._original = attention_cls._run_flash_attention
        capture = self

        def wrapped(
            attn_self,
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q,
            max_seqlen_k,
        ):
            if capture._original is None:
                raise RuntimeError("Flash oracle capture is not installed")
            call_index = capture._call_index
            capture._call_index += 1
            out = capture._original(
                attn_self,
                q,
                k,
                v,
                cu_q,
                cu_k,
                max_seqlen_q,
                max_seqlen_k,
            )
            if len(capture.calls) < capture.max_calls:
                window_size = attn_self._flash_window_size()
                source_context = (
                    int(attn_self.context)
                    if attn_self.context is not None and attn_self.causal
                    else None
                )
                try:
                    ref = _sdpa_reference_from_varlen(
                        q,
                        k,
                        v,
                        cu_q,
                        cu_k,
                        max_seqlen_q=int(max_seqlen_q),
                        causal=bool(attn_self.causal),
                        window_size=window_size,
                        source_context=source_context,
                    )
                    comparison = _tensor_comparison(
                        ref.detach().to("cpu"), out.detach().to("cpu")
                    )
                    error: str | None = None
                except Exception as exc:
                    comparison = None
                    error = repr(exc)
                capture.calls.append(
                    {
                        "call_index": call_index,
                        "attention_class": attn_self.source.__class__.__name__,
                        "q_shape": list(q.shape),
                        "k_shape": list(k.shape),
                        "v_shape": list(v.shape),
                        "out_shape": list(out.shape),
                        "cu_q": _to_jsonable(cu_q),
                        "cu_k": _to_jsonable(cu_k),
                        "max_seqlen_q": int(max_seqlen_q),
                        "max_seqlen_k": int(max_seqlen_k),
                        "causal": bool(attn_self.causal),
                        "window_size": list(window_size),
                        "source_context": source_context,
                        "comparison_to_sdpa": comparison,
                        "error": error,
                    }
                )
            return out

        attention_cls._run_flash_attention = wrapped
        return self

    def __exit__(self, *_: object) -> None:
        if self._original is not None:
            vocoder_decoder_module.MossTTSLocalAttention._run_flash_attention = (
                self._original
            )
            self._original = None

    def summary(self) -> dict[str, Any]:
        return {
            "captured_call_count": len(self.calls),
            "observed_call_count": self._call_index,
            "calls": self.calls,
        }


class _OwnedRouteCapture:
    """Record owned decoder routing decisions without changing execution."""

    def __init__(self) -> None:
        self.stage_routes: Counter[str] = Counter()
        self.attention_routes: Counter[str] = Counter()
        self.examples: dict[str, list[dict[str, Any]]] = {
            "stage": [],
            "attention": [],
        }
        self._original_stage_forward = None
        self._original_attention_forward = None

    def __enter__(self) -> "_OwnedRouteCapture":
        stage_cls = vocoder_decoder_module.MossTTSLocalProjectedTransformer
        attention_cls = vocoder_decoder_module.MossTTSLocalAttention
        self._original_stage_forward = stage_cls.forward
        self._original_attention_forward = attention_cls.forward
        capture = self

        def stage_forward(stage_self, x, input_lengths, **kwargs):
            source_streaming = bool(getattr(stage_self.source, "is_streaming", False))
            backend = stage_self.transformer.resolve_attention_implementation(
                x.transpose(1, 2)
            )
            if not source_streaming and backend != "flash_attention_2":
                route = "source_stage_delegate"
            elif not source_streaming and backend == "flash_attention_2":
                route = "owned_packed_stage"
            else:
                route = "owned_streaming_stage"
            capture.stage_routes[route] += 1
            capture._append_example(
                "stage",
                {
                    "route": route,
                    "source_class": stage_self.source.__class__.__name__,
                    "source_streaming": source_streaming,
                    "backend": backend,
                    "input_shape": list(x.shape),
                    "input_lengths": _small_tensor_list(input_lengths),
                },
            )
            return capture._original_stage_forward(
                stage_self, x, input_lengths, **kwargs
            )

        def attention_forward(attn_self, query, **kwargs):
            streaming_state = getattr(attn_self.source, "_streaming_state", None)
            backend = attn_self.resolve_attention_implementation(
                query,
                is_streaming=streaming_state is not None,
            )
            if streaming_state is not None:
                if backend == vocoder_decoder_module._SOURCE_ATTENTION:
                    route = "streaming_source_attention"
                elif backend == "flash_attention_2":
                    route = "streaming_flash_attention"
                else:
                    route = "streaming_owned_attention"
            elif backend == "flash_attention_2":
                route = "packed_flash_attention"
            else:
                route = "dense_source_attention"
            capture.attention_routes[route] += 1
            capture._append_example(
                "attention",
                {
                    "route": route,
                    "source_class": attn_self.source.__class__.__name__,
                    "source_streaming_state": streaming_state is not None,
                    "backend": backend,
                    "query_shape": list(query.shape),
                    "input_lengths": _small_tensor_list(kwargs.get("input_lengths")),
                },
            )
            return capture._original_attention_forward(attn_self, query, **kwargs)

        stage_cls.forward = stage_forward
        attention_cls.forward = attention_forward
        return self

    def __exit__(self, *_: object) -> None:
        stage_cls = vocoder_decoder_module.MossTTSLocalProjectedTransformer
        attention_cls = vocoder_decoder_module.MossTTSLocalAttention
        if self._original_stage_forward is not None:
            stage_cls.forward = self._original_stage_forward
            self._original_stage_forward = None
        if self._original_attention_forward is not None:
            attention_cls.forward = self._original_attention_forward
            self._original_attention_forward = None

    def _append_example(self, kind: str, value: dict[str, Any]) -> None:
        examples = self.examples[kind]
        if len(examples) < 12:
            examples.append(value)

    def summary(self) -> dict[str, Any]:
        return {
            "stage_routes": dict(self.stage_routes),
            "attention_routes": dict(self.attention_routes),
            "examples": self.examples,
        }


def _small_tensor_list(value: Any) -> list[int] | None:
    if not isinstance(value, torch.Tensor):
        return None
    if value.numel() > 32:
        return None
    return [int(item) for item in value.detach().to("cpu").reshape(-1).tolist()]


def _parse_probe(value: str) -> tuple[int, int]:
    try:
        batch, frames = value.lower().split("x", 1)
        batch_size = int(batch)
        frame_count = int(frames)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"probe must use BxT form, for example 8x100; got {value!r}"
        ) from exc
    if batch_size < 1 or frame_count < 1:
        raise argparse.ArgumentTypeError(
            f"probe batch and frames must be positive, got {value!r}"
        )
    return batch_size, frame_count


def _device_of_processor(processor: Any, fallback: str) -> torch.device:
    codec = getattr(processor, "audio_tokenizer", None)
    if codec is not None:
        try:
            return next(codec.parameters()).device
        except StopIteration:
            pass
    return torch.device(fallback)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_sdp_settings() -> dict[str, bool | None]:
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return {
            "cudnn_sdp": None,
            "flash_sdp": None,
            "math_sdp": None,
            "mem_efficient_sdp": None,
        }

    def _enabled(name: str) -> bool | None:
        fn = getattr(cuda_backend, name, None)
        return bool(fn()) if callable(fn) else None

    return {
        "cudnn_sdp": _enabled("cudnn_sdp_enabled"),
        "flash_sdp": _enabled("flash_sdp_enabled"),
        "math_sdp": _enabled("math_sdp_enabled"),
        "mem_efficient_sdp": _enabled("mem_efficient_sdp_enabled"),
    }


def _set_cuda_sdp_backend(backend: str) -> None:
    if backend == "default":
        return
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        raise RuntimeError("torch.backends.cuda is unavailable")
    setters = {
        "cudnn": getattr(cuda_backend, "enable_cudnn_sdp", None),
        "flash": getattr(cuda_backend, "enable_flash_sdp", None),
        "math": getattr(cuda_backend, "enable_math_sdp", None),
        "efficient": getattr(cuda_backend, "enable_mem_efficient_sdp", None),
    }
    missing = [name for name, setter in setters.items() if not callable(setter)]
    if missing:
        raise RuntimeError(
            "this PyTorch build cannot select CUDA SDPA backend; missing " f"{missing}"
        )
    for name, setter in setters.items():
        setter(name == backend)


def _audio_vocab_size(processor: Any) -> int:
    for owner in (
        getattr(processor, "model_config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "config", None),
    ):
        if owner is None:
            continue
        for attr in ("audio_vocab_size", "vocab_size", "codebook_size"):
            value = getattr(owner, attr, None)
            if value:
                return int(value)
    return 1024


def _num_codebooks(processor: Any) -> int:
    value = getattr(getattr(processor, "model_config", None), "n_vq", None)
    if value is not None:
        return int(value)
    value = getattr(getattr(processor, "audio_tokenizer", None), "num_codebooks", None)
    if value is not None:
        return int(value)
    return 12


def _make_codes(
    *,
    batch_size: int,
    frames: int,
    n_vq: int,
    vocab_size: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randint(0, vocab_size, (frames, n_vq), generator=generator)
        for _ in range(batch_size)
    ]


def _module_list(value: Any) -> list[nn.Module]:
    if isinstance(value, nn.ModuleList):
        return list(value)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, nn.Module) for item in value
    ):
        return list(value)
    return []


def _decoder_stages(decoder: Any) -> list[nn.Module]:
    stages = getattr(decoder, "stages", None)
    if stages is not None:
        out = _module_list(stages)
        if out:
            return out
    return _module_list(decoder)


def _add_if_module(
    modules: list[tuple[str, nn.Module]],
    name: str,
    value: Any,
) -> None:
    if isinstance(value, nn.Module):
        modules.append((name, value))


def _trace_modules(decoder: Any, *, trace_level: str) -> list[tuple[str, nn.Module]]:
    modules: list[tuple[str, nn.Module]] = []
    if trace_level == "none":
        return modules

    for stage_index, stage in enumerate(_decoder_stages(decoder)):
        stage_name = f"decoder.stage_{stage_index:02d}"
        modules.append((stage_name, stage))
        if trace_level == "stage":
            continue

        _add_if_module(
            modules,
            f"{stage_name}.input_proj",
            getattr(stage, "input_proj", None),
        )
        transformer = getattr(stage, "transformer", None)
        _add_if_module(modules, f"{stage_name}.transformer", transformer)
        _add_if_module(
            modules,
            f"{stage_name}.output_proj",
            getattr(stage, "output_proj", None),
        )
        if trace_level == "layer" or transformer is None:
            if transformer is not None:
                for layer_index, layer in enumerate(
                    _module_list(getattr(transformer, "layers", None))
                ):
                    modules.append(
                        (f"{stage_name}.transformer.layer_{layer_index:02d}", layer)
                    )
            continue

        for layer_index, layer in enumerate(
            _module_list(getattr(transformer, "layers", None))
        ):
            layer_name = f"{stage_name}.transformer.layer_{layer_index:02d}"
            modules.append((layer_name, layer))
            _add_if_module(
                modules,
                f"{layer_name}.norm1",
                getattr(layer, "norm1", None),
            )
            self_attn = getattr(layer, "self_attn", None)
            _add_if_module(modules, f"{layer_name}.self_attn", self_attn)
            if self_attn is not None:
                _add_if_module(
                    modules,
                    f"{layer_name}.self_attn.in_proj",
                    getattr(self_attn, "in_proj", None),
                )
                _add_if_module(
                    modules,
                    f"{layer_name}.self_attn.out_proj",
                    getattr(self_attn, "out_proj", None),
                )
            _add_if_module(
                modules,
                f"{layer_name}.layer_scale_1",
                getattr(layer, "layer_scale_1", None),
            )
            _add_if_module(
                modules,
                f"{layer_name}.norm2",
                getattr(layer, "norm2", None),
            )
            ffn = getattr(layer, "ffn", None)
            _add_if_module(modules, f"{layer_name}.ffn", ffn)
            if isinstance(ffn, nn.Sequential):
                for ffn_index, ffn_module in enumerate(ffn):
                    modules.append((f"{layer_name}.ffn.{ffn_index}", ffn_module))
            _add_if_module(
                modules,
                f"{layer_name}.layer_scale_2",
                getattr(layer, "layer_scale_2", None),
            )
    return modules


def _tensor_comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    same_dtype = reference.dtype == candidate.dtype
    if not same_shape:
        return {
            "same_shape": False,
            "same_dtype": same_dtype,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    equal = bool(torch.equal(reference, candidate)) if same_dtype else False
    delta = reference.to(torch.float32) - candidate.to(torch.float32)
    abs_delta = delta.abs()
    signal = reference.to(torch.float32)
    signal_energy = float((signal * signal).sum().item())
    noise_energy = float((delta * delta).sum().item())
    if noise_energy == 0.0:
        snr_db: float | str = "inf"
    elif signal_energy == 0.0:
        snr_db = "-inf"
    else:
        snr_db = 10.0 * math.log10(signal_energy / noise_energy)
    return {
        "same_shape": True,
        "same_dtype": same_dtype,
        "torch_equal": equal,
        "max_abs_delta": float(abs_delta.max().item()) if abs_delta.numel() else 0.0,
        "mean_abs_delta": float(abs_delta.mean().item()) if abs_delta.numel() else 0.0,
        "snr_db": snr_db,
    }


def _compare_records(
    reference: OrderedDict[str, _CapturedTensor],
    candidate: OrderedDict[str, _CapturedTensor],
) -> dict[str, Any]:
    common = []
    first_mismatch: dict[str, Any] | None = None
    first_uncompared: dict[str, Any] | None = None

    for key, ref_record in reference.items():
        cand_record = candidate.get(key)
        if cand_record is None:
            first_mismatch = {
                "kind": "missing_candidate_record",
                "key": key,
                "reference": ref_record.summary(),
            }
            break
        common.append(key)
        if ref_record.tensor is None or cand_record.tensor is None:
            if first_uncompared is None:
                first_uncompared = {
                    "kind": "tensor_not_stored",
                    "key": key,
                    "reference": ref_record.summary(),
                    "candidate": cand_record.summary(),
                }
            continue
        comparison = _tensor_comparison(ref_record.tensor, cand_record.tensor)
        if not comparison.get("torch_equal", False):
            first_mismatch = {
                "kind": "tensor_mismatch",
                "key": key,
                "reference": ref_record.summary(),
                "candidate": cand_record.summary(),
                "comparison": comparison,
            }
            break

    extra_candidate = [key for key in candidate if key not in reference]
    return {
        "reference_record_count": len(reference),
        "candidate_record_count": len(candidate),
        "common_record_count": len(common),
        "missing_candidate_count": (
            len(reference) - len(common)
            if first_mismatch and first_mismatch["kind"] == "missing_candidate_record"
            else 0
        ),
        "extra_candidate_count": len(extra_candidate),
        "first_mismatch": first_mismatch,
        "first_uncompared": first_uncompared,
    }


def _compare_outputs(
    reference: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> dict[str, Any]:
    comparisons = []
    for index, (ref, cand) in enumerate(zip(reference, candidate)):
        comparisons.append({"index": index, **_tensor_comparison(ref, cand)})
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "all_equal": len(reference) == len(candidate)
        and all(item.get("torch_equal", False) for item in comparisons),
        "max_abs_delta": max(
            (float(item.get("max_abs_delta", 0.0)) for item in comparisons),
            default=0.0,
        ),
        "mean_abs_delta_max": max(
            (float(item.get("mean_abs_delta", 0.0)) for item in comparisons),
            default=0.0,
        ),
        "comparisons": comparisons,
    }


def _decode_processor(
    processor: Any, codes_list: list[torch.Tensor]
) -> list[torch.Tensor]:
    return [
        torch.as_tensor(wav).detach().to("cpu")
        for wav in processor.decode_audio_codes(codes_list)
    ]


def _decode_owned(
    processor: Any,
    codes_list: list[torch.Tensor],
    owned_decoder: nn.Module,
) -> list[torch.Tensor]:
    codec = processor.audio_tokenizer
    with use_moss_tts_local_vocoder_decoder(codec, owned_decoder):
        return [
            torch.as_tensor(wav).detach().to("cpu")
            for wav in processor.decode_audio_codes(codes_list)
        ]


def _run_with_trace(
    *,
    label: str,
    decoder: Any,
    decode_fn,
    device: torch.device,
    trace_level: str,
    trace_sdpa: bool,
    trace_owned_routes: bool,
    trace_flash_oracle: bool,
    flash_oracle_max_calls: int,
    max_elements_per_tensor: int,
) -> tuple[
    list[torch.Tensor],
    OrderedDict[str, _CapturedTensor],
    OrderedDict[str, _CapturedTensor],
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    modules = _trace_modules(decoder, trace_level=trace_level)
    capture = _TraceCapture(max_elements_per_tensor=max_elements_per_tensor)
    sdpa_capture = (
        _SdpaCapture(max_elements_per_tensor=max_elements_per_tensor)
        if trace_sdpa
        else None
    )
    route_capture = _OwnedRouteCapture() if trace_owned_routes else None
    flash_oracle_capture = (
        _FlashAttentionOracleCapture(max_calls=flash_oracle_max_calls)
        if trace_flash_oracle
        else None
    )
    logger.info("%s: installing %d trace hooks", label, len(modules) * 2)
    capture.install(modules)
    try:
        _synchronize(device)
        sdpa_context = sdpa_capture if sdpa_capture is not None else nullcontext()
        route_context = route_capture if route_capture is not None else nullcontext()
        flash_oracle_context = (
            flash_oracle_capture if flash_oracle_capture is not None else nullcontext()
        )
        with route_context, sdpa_context, flash_oracle_context, torch.no_grad():
            outputs = decode_fn()
        _synchronize(device)
    finally:
        capture.close()
    sdpa_records = sdpa_capture.records if sdpa_capture is not None else OrderedDict()
    route_summary = route_capture.summary() if route_capture is not None else None
    flash_oracle_summary = (
        flash_oracle_capture.summary() if flash_oracle_capture is not None else None
    )
    logger.info(
        "%s: captured %d tensor records, %d SDPA records, and %d flash oracle calls",
        label,
        len(capture.records),
        len(sdpa_records),
        (
            int(flash_oracle_summary["captured_call_count"])
            if flash_oracle_summary is not None
            else 0
        ),
    )
    return outputs, capture.records, sdpa_records, route_summary, flash_oracle_summary


def _attention_global_summary(codec: Any) -> dict[str, Any]:
    decoder = getattr(codec, "decoder", None)
    modules = list(decoder.modules()) if hasattr(decoder, "modules") else []
    attention_modules = [
        module
        for module in modules
        if module.__class__.__name__ == "MossAudioTokenizerMultiheadAttention"
    ]
    python_modules: dict[str, ModuleType] = {}
    for module in attention_modules:
        loaded = sys.modules.get(module.__class__.__module__)
        if loaded is not None:
            python_modules[module.__class__.__module__] = loaded
    globals_summary = {}
    for name, module in python_modules.items():
        globals_summary[name] = {
            "HAS_FLASH_ATTN": bool(getattr(module, "HAS_FLASH_ATTN", False)),
            "flash_attn_varlen_func": repr(
                getattr(module, "flash_attn_varlen_func", None)
            ),
            "file": getattr(module, "__file__", None),
        }
    implementation_counts: dict[str, int] = {}
    for module in attention_modules:
        implementation = str(getattr(module, "attention_implementation", "unknown"))
        implementation_counts[implementation] = (
            implementation_counts.get(implementation, 0) + 1
        )
    return {
        "attention_module_count": len(attention_modules),
        "attention_implementation_counts": implementation_counts,
        "python_module_globals": globals_summary,
    }


def _dependency_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import sglang

        summary["sglang"] = getattr(sglang, "__version__", "unknown")
    except Exception as exc:
        summary["sglang_error"] = repr(exc)
    try:
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func

        summary["sglang_flash_attn_varlen_func"] = repr(flash_attn_varlen_func)
    except Exception as exc:
        summary["sglang_flash_attn_varlen_func_error"] = repr(exc)
    return summary


def _run_probe(
    processor: Any,
    *,
    batch_size: int,
    frames: int,
    seed: int,
    candidates: list[str],
    trace_level: str,
    trace_sdpa: bool,
    trace_owned_routes: bool,
    trace_flash_oracle: bool,
    flash_oracle_max_calls: int,
    max_elements_per_tensor: int,
    device: torch.device,
    dump_first_mismatch: bool,
    output_dir: Path,
) -> dict[str, Any]:
    n_vq = _num_codebooks(processor)
    vocab_size = _audio_vocab_size(processor)
    codes_list = _make_codes(
        batch_size=batch_size,
        frames=frames,
        n_vq=n_vq,
        vocab_size=vocab_size,
        seed=seed,
    )
    codec = processor.audio_tokenizer
    owned_decoder = build_moss_tts_local_vocoder_decoder(
        codec,
        max_batch_size=batch_size,
        max_chunk_frames=frames,
    )
    if hasattr(owned_decoder, "eval"):
        owned_decoder.eval()

    source_outputs, source_records, source_sdpa_records, _, _ = _run_with_trace(
        label="processor",
        decoder=codec.decoder,
        decode_fn=lambda: _decode_processor(processor, codes_list),
        device=device,
        trace_level=trace_level,
        trace_sdpa=trace_sdpa,
        trace_owned_routes=False,
        trace_flash_oracle=False,
        flash_oracle_max_calls=0,
        max_elements_per_tensor=max_elements_per_tensor,
    )

    candidate_reports: dict[str, Any] = {}
    for candidate in candidates:
        if candidate == "owned":
            candidate_decoder = owned_decoder
            candidate_decode = lambda: _decode_owned(
                processor, codes_list, owned_decoder
            )
        elif candidate == "processor-repeat":
            candidate_decoder = codec.decoder
            candidate_decode = lambda: _decode_processor(processor, codes_list)
        else:
            raise ValueError(f"unsupported candidate {candidate!r}")

        (
            candidate_outputs,
            candidate_records,
            candidate_sdpa_records,
            route_summary,
            flash_oracle_summary,
        ) = _run_with_trace(
            label=candidate,
            decoder=candidate_decoder,
            decode_fn=candidate_decode,
            device=device,
            trace_level=trace_level,
            trace_sdpa=trace_sdpa,
            trace_owned_routes=trace_owned_routes and candidate == "owned",
            trace_flash_oracle=trace_flash_oracle and candidate == "owned",
            flash_oracle_max_calls=flash_oracle_max_calls,
            max_elements_per_tensor=max_elements_per_tensor,
        )
        record_comparison = _compare_records(source_records, candidate_records)
        sdpa_comparison = _compare_records(source_sdpa_records, candidate_sdpa_records)
        output_comparison = _compare_outputs(source_outputs, candidate_outputs)
        first_mismatch = record_comparison.get("first_mismatch")
        dump_path = None
        if dump_first_mismatch and first_mismatch and first_mismatch.get("key"):
            key = str(first_mismatch["key"])
            source = source_records.get(key)
            candidate_record = candidate_records.get(key)
            if (
                source is not None
                and candidate_record is not None
                and source.tensor is not None
            ):
                dump_path = output_dir / (
                    f"first_mismatch_b{batch_size}_t{frames}_{candidate}_"
                    f"{key.replace('/', '_').replace('#', '_')}.pt"
                )
                torch.save(
                    {
                        "key": key,
                        "source": source.tensor,
                        "candidate": candidate_record.tensor,
                        "comparison": first_mismatch.get("comparison"),
                    },
                    dump_path,
                )
        first_sdpa_mismatch = sdpa_comparison.get("first_mismatch")
        sdpa_dump_path = None
        if (
            dump_first_mismatch
            and first_sdpa_mismatch
            and first_sdpa_mismatch.get("key")
        ):
            key = str(first_sdpa_mismatch["key"])
            source = source_sdpa_records.get(key)
            candidate_record = candidate_sdpa_records.get(key)
            if (
                source is not None
                and candidate_record is not None
                and source.tensor is not None
            ):
                sdpa_dump_path = output_dir / (
                    f"first_sdpa_mismatch_b{batch_size}_t{frames}_{candidate}_"
                    f"{key.replace('/', '_').replace('#', '_')}.pt"
                )
                torch.save(
                    {
                        "key": key,
                        "source": source.tensor,
                        "candidate": candidate_record.tensor,
                        "comparison": first_sdpa_mismatch.get("comparison"),
                    },
                    sdpa_dump_path,
                )

        logger.info(
            "probe b=%d t=%d candidate=%s output_equal=%s max_abs=%.6g "
            "first_mismatch=%s first_sdpa_mismatch=%s",
            batch_size,
            frames,
            candidate,
            output_comparison["all_equal"],
            output_comparison["max_abs_delta"],
            first_mismatch.get("key") if first_mismatch else None,
            first_sdpa_mismatch.get("key") if first_sdpa_mismatch else None,
        )
        candidate_reports[candidate] = {
            "output_comparison": output_comparison,
            "record_comparison": record_comparison,
            "sdpa_comparison": sdpa_comparison,
            "flash_oracle": flash_oracle_summary,
            "owned_route_summary": route_summary,
            "first_mismatch_dump": str(dump_path) if dump_path is not None else None,
            "first_sdpa_mismatch_dump": (
                str(sdpa_dump_path) if sdpa_dump_path is not None else None
            ),
        }

    legacy_owned = candidate_reports.get("owned")
    if legacy_owned is None:
        first_candidate = next(iter(candidate_reports.values()), None)
        legacy_owned = first_candidate or {
            "output_comparison": {},
            "record_comparison": {},
            "first_mismatch_dump": None,
            "first_sdpa_mismatch_dump": None,
            "owned_route_summary": None,
        }
    return {
        "batch_size": batch_size,
        "frames": frames,
        "seed": seed,
        "codebooks": n_vq,
        "vocab_size": vocab_size,
        "trace_level": trace_level,
        "trace_sdpa": trace_sdpa,
        "trace_owned_routes": trace_owned_routes,
        "trace_flash_oracle": trace_flash_oracle,
        "flash_oracle_max_calls": flash_oracle_max_calls,
        "max_elements_per_tensor": max_elements_per_tensor,
        "candidates": candidate_reports,
        # Kept for older report consumers while this script is actively used.
        "output_comparison": legacy_owned["output_comparison"],
        "record_comparison": legacy_owned["record_comparison"],
        "first_mismatch_dump": legacy_owned["first_mismatch_dump"],
        "first_sdpa_mismatch_dump": legacy_owned.get("first_sdpa_mismatch_dump"),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# MOSS-TTS Local Vocoder Exactness",
        "",
        f"- schema: `{report['schema']}`",
        f"- model: `{report['model_path']}`",
        f"- device: `{report['device']}`",
        f"- trace level: `{report['trace_level']}`",
        f"- trace SDPA: `{report['trace_sdpa']}`",
        f"- trace owned routes: `{report['trace_owned_routes']}`",
        f"- trace flash oracle: `{report['trace_flash_oracle']}`",
        f"- CUDA SDPA backend: `{report['cuda_sdp_backend']}`",
        "",
        "## Attention Globals",
        "",
        "```json",
        json.dumps(report["attention_globals"], indent=2, sort_keys=True),
        "```",
        "",
        "## Probes",
        "",
        "| probe | candidate | output equal | max abs | mean abs max | first traced mismatch | first SDPA mismatch | dump | SDPA dump |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for probe in report["probes"]:
        for candidate, candidate_report in probe["candidates"].items():
            out = candidate_report["output_comparison"]
            mismatch = candidate_report["record_comparison"].get("first_mismatch")
            sdpa_mismatch = candidate_report["sdpa_comparison"].get("first_mismatch")
            lines.append(
                "| {batch}x{frames} | {candidate} | {equal} | {max_abs:.6g} | {mean_abs:.6g} | {mismatch} | {sdpa_mismatch} | {dump} | {sdpa_dump} |".format(
                    batch=probe["batch_size"],
                    frames=probe["frames"],
                    candidate=candidate,
                    equal=out["all_equal"],
                    max_abs=float(out["max_abs_delta"]),
                    mean_abs=float(out["mean_abs_delta_max"]),
                    mismatch=mismatch.get("key") if mismatch else "",
                    sdpa_mismatch=sdpa_mismatch.get("key") if sdpa_mismatch else "",
                    dump=candidate_report.get("first_mismatch_dump") or "",
                    sdpa_dump=candidate_report.get("first_sdpa_mismatch_dump") or "",
                )
            )
            route_summary = candidate_report.get("owned_route_summary")
            if route_summary:
                lines.extend(
                    [
                        "",
                        f"### Routes {probe['batch_size']}x{probe['frames']} {candidate}",
                        "",
                        "```json",
                        json.dumps(route_summary, indent=2, sort_keys=True),
                        "```",
                    ]
                )
            flash_oracle = candidate_report.get("flash_oracle")
            if flash_oracle:
                calls = flash_oracle.get("calls") or []
                lines.extend(
                    [
                        "",
                        f"### Flash Oracle {probe['batch_size']}x{probe['frames']} {candidate}",
                        "",
                        f"- observed calls: `{flash_oracle.get('observed_call_count')}`",
                        f"- captured calls: `{flash_oracle.get('captured_call_count')}`",
                        "",
                    ]
                )
                if calls:
                    lines.extend(
                        [
                            "| call | q | k | max_q | max_k | window | equal | max abs | mean abs | SNR | error |",
                            "|---:|---|---|---:|---:|---|---:|---:|---:|---:|---|",
                        ]
                    )
                    for call in calls:
                        comparison = call.get("comparison_to_sdpa") or {}
                        lines.append(
                            "| {call} | {q} | {k} | {max_q} | {max_k} | {window} | {equal} | {max_abs:.6g} | {mean_abs:.6g} | {snr} | {error} |".format(
                                call=call.get("call_index"),
                                q=call.get("q_shape"),
                                k=call.get("k_shape"),
                                max_q=call.get("max_seqlen_q"),
                                max_k=call.get("max_seqlen_k"),
                                window=call.get("window_size"),
                                equal=comparison.get("torch_equal", False),
                                max_abs=float(comparison.get("max_abs_delta", 0.0)),
                                mean_abs=float(comparison.get("mean_abs_delta", 0.0)),
                                snr=comparison.get("snr_db", ""),
                                error=call.get("error") or "",
                            )
                        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If `first traced mismatch` is empty but output parity fails, rerun with "
            "`--trace-level module` and a higher `--max-elements-per-tensor`. If "
            "the first mismatch is in an owned wrapper boundary before any SGLang "
            "attention call, rerun with `--trace-sdpa` to determine whether the "
            "attention inputs, mask, or SDPA output first diverged.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        help="HF model id or local checkpoint path",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--probe",
        action="append",
        type=_parse_probe,
        default=[],
        help="probe shape in BxT form, e.g. 8x100; repeatable",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidate",
        action="append",
        choices=("owned", "processor-repeat"),
        default=[],
        help=(
            "candidate path to compare against the first processor run; repeatable. "
            "Use processor-repeat to test backend determinism."
        ),
    )
    parser.add_argument(
        "--cuda-sdpa-backend",
        choices=("default", "math", "flash", "efficient", "cudnn"),
        default="default",
        help="force a single PyTorch CUDA SDPA backend before loading the model",
    )
    parser.add_argument(
        "--trace-level",
        choices=("none", "stage", "layer", "module"),
        default="module",
        help="decoder boundary detail to capture",
    )
    parser.add_argument(
        "--trace-sdpa",
        action="store_true",
        help="also capture torch.nn.functional.scaled_dot_product_attention tensors",
    )
    parser.add_argument(
        "--trace-owned-routes",
        action="store_true",
        help="record owned vocoder stage and attention routing decisions",
    )
    parser.add_argument(
        "--trace-flash-oracle",
        action="store_true",
        help=(
            "wrap owned SGLang FlashAttention calls and compare them against "
            "a segment-wise PyTorch SDPA oracle built from the same packed Q/K/V"
        ),
    )
    parser.add_argument(
        "--flash-oracle-max-calls",
        type=int,
        default=1,
        help="maximum owned FlashAttention calls to compare per candidate/probe",
    )
    parser.add_argument(
        "--max-elements-per-tensor",
        type=int,
        default=10_000_000,
        help="summarize larger hook tensors without storing payloads",
    )
    parser.add_argument("--dump-first-mismatch", action="store_true")
    parser.add_argument("--no-markdown", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_arg_parser().parse_args()
    if args.max_elements_per_tensor < 1:
        raise SystemExit("--max-elements-per-tensor must be positive")
    if args.flash_oracle_max_calls < 1:
        raise SystemExit("--flash-oracle-max-calls must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_cuda_sdp_backend(args.cuda_sdpa_backend)
    logger.info("Loading MOSS-TTS Local processor from %s", args.model)
    processor = _load_moss_tts_local_processor(args.model, device=args.device)
    device = _device_of_processor(processor, args.device)
    codec = processor.audio_tokenizer

    probes = args.probe or [(1, 25), (1, 100), (1, 300), (8, 100), (8, 300)]
    candidates = args.candidate or ["owned"]
    report: dict[str, Any] = {
        "schema": "moss_tts_local_vocoder_exactness_v1",
        "model_path": args.model,
        "device": str(device),
        "trace_level": args.trace_level,
        "trace_sdpa": args.trace_sdpa,
        "trace_owned_routes": args.trace_owned_routes,
        "trace_flash_oracle": args.trace_flash_oracle,
        "flash_oracle_max_calls": args.flash_oracle_max_calls,
        "cuda_sdp_backend": args.cuda_sdpa_backend,
        "cuda_sdp_settings": _cuda_sdp_settings(),
        "candidates": candidates,
        "dependencies": _dependency_summary(),
        "attention_globals": _attention_global_summary(codec),
        "probes": [],
    }
    for index, (batch_size, frames) in enumerate(probes):
        logger.info("Running exactness probe %dx%d", batch_size, frames)
        report["probes"].append(
            _run_probe(
                processor,
                batch_size=batch_size,
                frames=frames,
                seed=args.seed + index,
                candidates=candidates,
                trace_level=args.trace_level,
                trace_sdpa=args.trace_sdpa,
                trace_owned_routes=args.trace_owned_routes,
                trace_flash_oracle=args.trace_flash_oracle,
                flash_oracle_max_calls=args.flash_oracle_max_calls,
                max_elements_per_tensor=args.max_elements_per_tensor,
                device=device,
                dump_first_mismatch=args.dump_first_mismatch,
                output_dir=output_dir,
            )
        )

    json_path = output_dir / "moss_tts_local_vocoder_exactness.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %s", json_path)
    if not args.no_markdown:
        markdown_path = output_dir / "moss_tts_local_vocoder_exactness.md"
        _write_markdown(report, markdown_path)
        logger.info("Wrote %s", markdown_path)


if __name__ == "__main__":
    main()
