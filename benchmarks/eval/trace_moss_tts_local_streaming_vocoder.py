# SPDX-License-Identifier: Apache-2.0
"""Trace MOSS-TTS Local streaming vocoder attention semantics.

This is a development harness for the streaming SGLang-vocoder work. It runs
the real MOSS-Audio-Tokenizer-v2 codec through the same persistent
``codec.streaming(B)`` session contract used by serving, records attention
module/SDPA call shapes, and optionally compares each captured SDPA call against
SGLang varlen FlashAttention when the call can be represented as the MOSS
streaming local-causal mask.

Example:

    PYTHONPATH=. python -m benchmarks.eval.trace_moss_tts_local_streaming_vocoder \
        --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
        --device cuda:0 \
        --output-dir /data/moss_streaming_vocoder_trace
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.models.moss_tts_local.audio_tokenizer import (
    load_moss_tts_local_audio_tokenizer,
)
from sglang_omni.models.moss_tts_local.stages import (
    _load_moss_tts_local_processor,
    _resolve_audio_tokenizer_model_path,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import _CodecStreamSession

logger = logging.getLogger(__name__)


_DEFAULT_BOUNDARY_FRAMES = [1, 4, 5, 8, 9, 10, 11, 12, 13, 20, 22, 24, 25, 100]


@dataclass(frozen=True)
class _StepSpec:
    name: str
    slots: dict[int, int]
    release_after: tuple[int, ...] = ()


@dataclass(frozen=True)
class _CaseSpec:
    name: str
    steps: tuple[_StepSpec, ...]


@dataclass(frozen=True)
class _PackedStreamingLocalAttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int
    context: int
    history_lengths: list[int]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
        help="MOSS-TTS Local model path used to resolve n_vq and codec path.",
    )
    parser.add_argument(
        "--codec-model",
        default=None,
        help="Override MOSS-Audio-Tokenizer-v2 checkpoint path.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stream-slots", type=int, default=8)
    parser.add_argument("--offline-slots", type=int, default=8)
    parser.add_argument("--max-step-frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--disable-cudnn-sdp",
        action="store_true",
        help="Disable cuDNN SDPA in-process for hosts with cuDNN execution-plan failures.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help=(
            "Optional case selector. Built-ins: boundary, multi, staggered, "
            "reuse, offline, long. May be repeated. Default: all."
        ),
    )
    parser.add_argument(
        "--disable-oracle",
        action="store_true",
        help="Skip SGLang FlashAttention oracle comparisons for SDPA calls.",
    )
    parser.add_argument(
        "--state-sample-limit",
        type=int,
        default=16,
        help="Per-step number of streaming-state modules to summarize.",
    )
    parser.add_argument(
        "--state-sample-root",
        default="decoder",
        help="Streaming-state module prefix to sample per step, or 'all'.",
    )
    parser.add_argument(
        "--mask-structure",
        action="store_true",
        help="Record compact bool-mask contiguity/range summaries for SDPA calls.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _cuda_sdp_settings() -> dict[str, bool | None]:
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        return {
            "cudnn_sdp": None,
            "flash_sdp": None,
            "math_sdp": None,
            "mem_efficient_sdp": None,
        }

    def enabled(name: str) -> bool | None:
        fn = getattr(cuda_backend, name, None)
        return bool(fn()) if callable(fn) else None

    return {
        "cudnn_sdp": enabled("cudnn_sdp_enabled"),
        "flash_sdp": enabled("flash_sdp_enabled"),
        "math_sdp": enabled("math_sdp_enabled"),
        "mem_efficient_sdp": enabled("mem_efficient_sdp_enabled"),
    }


def _disable_cudnn_sdp() -> None:
    fn = getattr(getattr(torch.backends, "cuda", None), "enable_cudnn_sdp", None)
    if callable(fn):
        fn(False)


def _dependency_summary(device: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev = torch.device(device)
        summary["device_name"] = torch.cuda.get_device_name(dev)
        summary["device_capability"] = list(torch.cuda.get_device_capability(dev))
    try:
        import sglang

        summary["sglang"] = getattr(sglang, "__version__", "unknown")
    except Exception as exc:
        summary["sglang"] = f"unavailable: {type(exc).__name__}: {exc}"
    return summary


def _attention_implementation_summary(codec: Any) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    module_count = 0
    for module in codec.modules():
        implementation = getattr(module, "attention_implementation", None)
        if implementation is not None:
            counts[str(implementation)] += 1
            module_count += 1

    globals_summary = {}
    for name, module in sorted(sys.modules.items()):
        if "moss" not in name.lower() or "tokenizer" not in name.lower():
            continue
        has_flash = getattr(module, "HAS_FLASH_ATTN", None)
        flash_func = getattr(module, "flash_attn_varlen_func", None)
        if has_flash is None and flash_func is None:
            continue
        globals_summary[name] = {
            "file": getattr(module, "__file__", None),
            "HAS_FLASH_ATTN": has_flash,
            "flash_attn_varlen_func": "None" if flash_func is None else str(flash_func),
        }

    return {
        "attention_module_count": module_count,
        "attention_implementation_counts": dict(counts),
        "python_module_globals": globals_summary,
    }


def _tensor_summary(
    tensor: torch.Tensor, *, include_values: bool = False
) -> dict[str, Any]:
    detached = tensor.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "stride": list(detached.stride()),
        "contiguous": bool(detached.is_contiguous()),
        "last_dim_contiguous": bool(detached.ndim == 0 or detached.stride(-1) == 1),
    }
    if detached.numel() == 0:
        summary["nan"] = False
        return summary
    if detached.is_floating_point() or detached.is_complex():
        finite = torch.isfinite(detached)
        summary["nan"] = bool(torch.isnan(detached).any().item())
        summary["finite"] = bool(finite.all().item())
        if bool(finite.any().item()):
            values = detached[finite].to("cpu", torch.float32)
            summary["min"] = float(values.min().item())
            summary["max"] = float(values.max().item())
            summary["mean_abs"] = float(values.abs().mean().item())
    elif detached.dtype == torch.bool:
        true_count = int(detached.sum().item())
        summary["true"] = true_count
        summary["false"] = int(detached.numel()) - true_count
    else:
        cpu = detached.to("cpu")
        summary["min"] = int(cpu.min().item())
        summary["max"] = int(cpu.max().item())
    if include_values and detached.numel() <= 64:
        summary["values"] = detached.to("cpu").tolist()
    return summary


def _object_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value, include_values=value.ndim <= 1)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_object_summary(item) for item in value[:16]]
    if isinstance(value, dict):
        return {str(k): _object_summary(v) for k, v in list(value.items())[:32]}
    return type(value).__name__


def _compare_tensors(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    result: dict[str, Any] = {
        "same_shape": same_shape,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if not same_shape:
        return result
    ref = reference.detach().to(torch.float32)
    out = candidate.detach().to(torch.float32)
    delta = (ref - out).abs()
    result.update(
        {
            "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
            "torch_equal": bool(torch.equal(reference, candidate)),
        }
    )
    noise = float(torch.sum((ref - out) ** 2).item())
    signal = float(torch.sum(ref**2).item())
    result["snr_db"] = "inf" if noise == 0.0 else 10.0 * math.log10(signal / noise)
    return result


def _bool_mask_structure(mask: torch.Tensor) -> dict[str, Any]:
    if mask.ndim != 4:
        return {"skipped": f"expected rank-4 bool mask, got rank {mask.ndim}"}

    with torch.no_grad():
        dense = mask.detach()
        batch, _, query_len, key_len = dense.shape
        mask_3d = dense[:, 0, :, :]
        true_per_batch = mask_3d.sum(dim=(1, 2))
        active_batches = torch.nonzero(true_per_batch > 0, as_tuple=False).flatten()
        true_per_query = mask_3d.sum(dim=-1)
        rows_with_true = true_per_query > 0

        result: dict[str, Any] = {
            "batch": int(batch),
            "query_len": int(query_len),
            "key_len": int(key_len),
            "active_batches": active_batches.to("cpu").tolist(),
            "true_per_batch": true_per_batch.to("cpu").tolist(),
            "rows_with_true": int(rows_with_true.sum().item()),
        }

        populated_counts = true_per_query[rows_with_true]
        if populated_counts.numel():
            result["true_per_query_min"] = int(populated_counts.min().item())
            result["true_per_query_max"] = int(populated_counts.max().item())

        # Global contiguity is cheap enough for normal streaming chunks. For very
        # large stage-10 masks, keep sampled row ranges and skip the full scan.
        if dense.numel() <= 50_000_000:
            transitions = (mask_3d[..., 1:] != mask_3d[..., :-1]).sum(dim=-1)
            noncontiguous = ((transitions > 2) & rows_with_true).sum()
            result["max_row_transitions"] = int(transitions.max().item())
            result["noncontiguous_rows"] = int(noncontiguous.item())
        else:
            result["contiguity_scan_skipped"] = f"mask numel {dense.numel()}"

        batch_samples = (
            active_batches[: min(4, active_batches.numel())].to("cpu").tolist()
        )
        if not batch_samples and batch:
            batch_samples = [0]
        query_samples = sorted({0, query_len // 2, max(0, query_len - 1)})
        rows = []
        for batch_index in batch_samples:
            for query_index in query_samples:
                row = mask_3d[batch_index, query_index]
                positions = torch.nonzero(row, as_tuple=False).flatten()
                count = int(positions.numel())
                if count:
                    first = int(positions[0].item())
                    last = int(positions[-1].item())
                    contiguous = (last - first + 1) == count
                else:
                    first = None
                    last = None
                    contiguous = True
                rows.append(
                    {
                        "batch": int(batch_index),
                        "query": int(query_index),
                        "count": count,
                        "first": first,
                        "last": last,
                        "contiguous": bool(contiguous),
                    }
                )
        result["sample_rows"] = rows
        return result


def _mask_summary(
    mask: Any, *, include_structure: bool = False
) -> dict[str, Any] | None:
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        return {"type": type(mask).__name__}
    summary = _tensor_summary(mask, include_values=mask.numel() <= 64)
    if include_structure and mask.dtype == torch.bool:
        summary["structure"] = _bool_mask_structure(mask)
    return summary


def _load_sglang_flash_attn_varlen_func() -> Any:
    from sglang.jit_kernel.flash_attention import flash_attn_varlen_func

    return flash_attn_varlen_func


def _derive_streaming_local_history_lengths(
    attn_mask: torch.Tensor,
    *,
    batch: int,
    query_len: int,
    key_len: int,
) -> tuple[int, list[int]]:
    if attn_mask.dtype != torch.bool:
        raise ValueError(f"expected bool attention mask, got {attn_mask.dtype}")
    if attn_mask.shape != (batch, 1, query_len, key_len):
        raise ValueError(
            "expected streaming mask shape [B, 1, Q, K], "
            f"got {tuple(attn_mask.shape)} for Q={query_len}, K={key_len}"
        )
    if key_len < query_len:
        raise ValueError(f"expected K >= Q, got Q={query_len}, K={key_len}")

    context = key_len - query_len
    q0 = attn_mask[:, 0, 0, :]
    positions = torch.arange(key_len, device=attn_mask.device).view(1, key_len)
    first = torch.where(q0, positions, key_len).amin(dim=1)
    if bool((first == key_len).any().item()):
        raise ValueError("streaming local mask has an empty first query row")

    history = context - first
    if bool(((history < 0) | (history > context)).any().item()):
        raise ValueError(
            f"derived invalid history lengths for context={context}: "
            f"{history.to('cpu').tolist()}"
        )
    return context, [int(value) for value in history.to("cpu").tolist()]


def _pack_streaming_local_attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor,
) -> _PackedStreamingLocalAttentionInputs:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("expected q/k/v rank 4 tensors")

    batch, heads, query_len, head_dim = query.shape
    if key.shape[:2] != (batch, heads) or value.shape[:2] != (batch, heads):
        raise ValueError(
            "q/k/v batch or head dimensions differ: "
            f"q={tuple(query.shape)}, k={tuple(key.shape)}, v={tuple(value.shape)}"
        )
    key_len = int(key.shape[2])
    if value.shape[2] != key_len or value.shape[3] != head_dim:
        raise ValueError(
            "k/v sequence or head dimensions differ: "
            f"k={tuple(key.shape)}, v={tuple(value.shape)}"
        )

    context, history_lengths = _derive_streaming_local_history_lengths(
        attn_mask,
        batch=batch,
        query_len=query_len,
        key_len=key_len,
    )

    q_chunks = []
    k_chunks = []
    v_chunks = []
    cu_q = [0]
    cu_k = [0]
    for batch_index, history in enumerate(history_lengths):
        key_start = context - history
        q_chunks.append(query[batch_index].transpose(0, 1).contiguous())
        k_chunks.append(key[batch_index, :, key_start:, :].transpose(0, 1).contiguous())
        v_chunks.append(
            value[batch_index, :, key_start:, :].transpose(0, 1).contiguous()
        )
        cu_q.append(cu_q[-1] + query_len)
        cu_k.append(cu_k[-1] + history + query_len)

    return _PackedStreamingLocalAttentionInputs(
        q=torch.cat(q_chunks, dim=0),
        k=torch.cat(k_chunks, dim=0),
        v=torch.cat(v_chunks, dim=0),
        cu_q=torch.tensor(cu_q, dtype=torch.int32, device=query.device),
        cu_k=torch.tensor(cu_k, dtype=torch.int32, device=query.device),
        max_q=query_len,
        max_k=max(history_lengths) + query_len if history_lengths else query_len,
        context=context,
        history_lengths=history_lengths,
    )


def _pad_streaming_local_attention_output(
    output: torch.Tensor,
    *,
    batch: int,
    heads: int,
    query_len: int,
    head_dim: int,
) -> torch.Tensor:
    return output.view(batch, query_len, heads, head_dim).transpose(1, 2)


class _StreamingTraceRecorder:
    def __init__(
        self,
        codec: Any,
        *,
        capture_oracle: bool,
        state_sample_limit: int,
        state_sample_root: str,
        capture_mask_structure: bool,
    ) -> None:
        self._codec = codec
        self._capture_oracle = capture_oracle
        self._state_sample_limit = int(state_sample_limit)
        self._state_sample_root = state_sample_root
        self._capture_mask_structure = capture_mask_structure
        self._flash_attn_varlen = None
        self._flash_attn_import_error: str | None = None
        if capture_oracle:
            try:
                self._flash_attn_varlen = _load_sglang_flash_attn_varlen_func()
            except Exception as exc:
                self._flash_attn_import_error = f"{type(exc).__name__}: {exc}"
        self._original_sdpa = F.scaled_dot_product_attention
        self._hooks: list[Any] = []
        self._module_stack: list[str] = []
        self._current_case: str | None = None
        self._current_step: str | None = None
        self._step_index = -1
        self.module_records: list[dict[str, Any]] = []
        self.sdpa_records: list[dict[str, Any]] = []

    def __enter__(self) -> "_StreamingTraceRecorder":
        self._install_module_hooks()
        F.scaled_dot_product_attention = self._wrap_sdpa  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        F.scaled_dot_product_attention = self._original_sdpa  # type: ignore[assignment]
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @contextlib.contextmanager
    def step(self, case_name: str, step_index: int, step_name: str):
        previous = (self._current_case, self._current_step, self._step_index)
        self._current_case = case_name
        self._current_step = step_name
        self._step_index = step_index
        try:
            yield
        finally:
            self._current_case, self._current_step, self._step_index = previous

    def state_inventory(self) -> list[dict[str, Any]]:
        inventory = []
        for path, module in self._codec.named_modules():
            state = getattr(module, "_streaming_state", None)
            if state is None:
                continue
            inventory.append(
                {
                    "module": path,
                    "module_type": type(module).__name__,
                    "state_type": type(state).__name__,
                    "state": self._state_object_summary(state),
                }
            )
        return inventory

    def state_sample(self) -> list[dict[str, Any]]:
        sample = []
        for path, module in self._codec.named_modules():
            if self._state_sample_root != "all" and not path.startswith(
                self._state_sample_root
            ):
                continue
            if len(sample) >= self._state_sample_limit:
                break
            state = getattr(module, "_streaming_state", None)
            if state is None:
                continue
            sample.append(
                {
                    "module": path,
                    "state_type": type(state).__name__,
                    "state": self._state_object_summary(state),
                }
            )
        return sample

    def _state_object_summary(self, state: Any) -> dict[str, Any]:
        fields = {}
        for name, value in sorted(vars(state).items()):
            if name.startswith("__"):
                continue
            fields[name] = _object_summary(value)
        return fields

    def _install_module_hooks(self) -> None:
        for path, module in self._codec.named_modules():
            if not self._is_attention_like(module):
                continue
            self._hooks.append(module.register_forward_pre_hook(self._pre_hook(path)))
            self._hooks.append(module.register_forward_hook(self._post_hook(path)))

    @staticmethod
    def _is_attention_like(module: Any) -> bool:
        return (
            hasattr(module, "attention_implementation")
            or hasattr(module, "_update_streaming_cache")
            or (
                hasattr(module, "in_proj")
                and hasattr(module, "out_proj")
                and "attention" in type(module).__name__.lower()
            )
        )

    def _pre_hook(self, path: str):
        def hook(module: Any, args: tuple[Any, ...]) -> None:
            self._module_stack.append(path)
            self.module_records.append(
                {
                    "event": "module_pre",
                    "case": self._current_case,
                    "step": self._current_step,
                    "step_index": self._step_index,
                    "module": path,
                    "module_type": type(module).__name__,
                    "args": [_object_summary(arg) for arg in args[:4]],
                }
            )

        return hook

    def _post_hook(self, path: str):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            self.module_records.append(
                {
                    "event": "module_post",
                    "case": self._current_case,
                    "step": self._current_step,
                    "step_index": self._step_index,
                    "module": path,
                    "module_type": type(module).__name__,
                    "output": _object_summary(output),
                }
            )
            if self._module_stack and self._module_stack[-1] == path:
                self._module_stack.pop()

        return hook

    def _wrap_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *args,
        **kwargs,
    ):
        attn_mask = kwargs.get("attn_mask", args[0] if len(args) >= 1 else None)
        dropout_p = kwargs.get("dropout_p", args[1] if len(args) >= 2 else 0.0)
        is_causal = bool(kwargs.get("is_causal", args[2] if len(args) >= 3 else False))
        scale = kwargs.get("scale", None)
        record: dict[str, Any] = {
            "case": self._current_case,
            "step": self._current_step,
            "step_index": self._step_index,
            "module": self._module_stack[-1] if self._module_stack else None,
            "query": _tensor_summary(query),
            "key": _tensor_summary(key),
            "value": _tensor_summary(value),
            "attn_mask": _mask_summary(
                attn_mask,
                include_structure=self._capture_mask_structure,
            ),
            "dropout_p": float(dropout_p),
            "is_causal": is_causal,
            "scale": scale,
        }
        output = self._original_sdpa(query, key, value, *args, **kwargs)
        record["output"] = _tensor_summary(output)
        if self._capture_oracle:
            record["oracle"] = self._run_flash_oracle(
                query=query,
                key=key,
                value=value,
                output=output,
                attn_mask=attn_mask,
                is_causal=is_causal,
                scale=scale,
            )
        self.sdpa_records.append(record)
        return output

    def _run_flash_oracle(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_mask: Any,
        is_causal: bool,
        scale: Any,
    ) -> dict[str, Any]:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            return {"skipped": "expected q/k/v rank 4"}
        if not query.is_cuda:
            return {"skipped": "query is not CUDA"}
        if self._flash_attn_varlen is None:
            return {"error": f"import failed: {self._flash_attn_import_error}"}

        try:
            batch_size, heads, q_len, head_dim = query.shape
            k_len = int(key.shape[2])
            if attn_mask is None:
                q = (
                    query.transpose(1, 2)
                    .contiguous()
                    .view(batch_size * q_len, heads, head_dim)
                )
                k = (
                    key.transpose(1, 2)
                    .contiguous()
                    .view(batch_size * k_len, heads, head_dim)
                )
                v = (
                    value.transpose(1, 2)
                    .contiguous()
                    .view(batch_size * k_len, heads, head_dim)
                )
                cu_q = torch.arange(
                    0,
                    (batch_size + 1) * q_len,
                    q_len,
                    dtype=torch.int32,
                    device=query.device,
                )
                cu_k = torch.arange(
                    0,
                    (batch_size + 1) * k_len,
                    k_len,
                    dtype=torch.int32,
                    device=query.device,
                )
                oracle = self._flash_attn_varlen(
                    q,
                    k,
                    v,
                    cu_q,
                    cu_k,
                    max_seqlen_q=q_len,
                    max_seqlen_k=k_len,
                    softmax_scale=scale,
                    causal=is_causal,
                    window_size=(-1, -1),
                )
                oracle = oracle.view(batch_size, q_len, heads, head_dim).transpose(1, 2)
                return {
                    "mode": "dense_unmasked",
                    "comparison": _compare_tensors(output, oracle),
                }

            if is_causal:
                return {"skipped": "masked call unexpectedly set is_causal=True"}
            if not isinstance(attn_mask, torch.Tensor):
                return {"skipped": f"mask type {type(attn_mask).__name__}"}
            packed = _pack_streaming_local_attention_inputs(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
            )
            oracle = self._flash_attn_varlen(
                packed.q,
                packed.k,
                packed.v,
                packed.cu_q,
                packed.cu_k,
                max_seqlen_q=packed.max_q,
                max_seqlen_k=packed.max_k,
                softmax_scale=scale,
                causal=True,
                window_size=(max(packed.context - 1, 0), 0),
            )
            oracle = _pad_streaming_local_attention_output(
                oracle,
                batch=batch_size,
                heads=heads,
                query_len=q_len,
                head_dim=head_dim,
            )
            return {
                "mode": "streaming_local_mask",
                "context": packed.context,
                "history_lengths": packed.history_lengths,
                "max_seqlen_q": packed.max_q,
                "max_seqlen_k": packed.max_k,
                "window_size": (max(packed.context - 1, 0), 0),
                "comparison": _compare_tensors(output, oracle),
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


def _audio_vocab_size(processor: Any) -> int:
    value = getattr(getattr(processor, "model_config", None), "audio_vocab_size", None)
    return int(value or 1024)


def _num_codebooks(processor: Any) -> int:
    value = getattr(getattr(processor, "model_config", None), "n_vq", None)
    return int(value or 12)


def _make_codes(
    *,
    n_vq: int,
    frames: int,
    vocab_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(0, vocab_size, (n_vq, frames), generator=generator)


def _build_cases(stream_slots: int, selected: set[str] | None) -> list[_CaseSpec]:
    cases: list[_CaseSpec] = []

    def enabled(name: str) -> bool:
        return selected is None or name in selected

    if enabled("boundary"):
        for frames in _DEFAULT_BOUNDARY_FRAMES:
            cases.append(
                _CaseSpec(
                    name=f"single_{frames}",
                    steps=(_StepSpec(name=f"slot0_t{frames}", slots={0: frames}),),
                )
            )
    if enabled("multi"):
        for batch_size in (2, 4, min(stream_slots, 8)):
            if batch_size < 2:
                continue
            slots = {slot: 25 for slot in range(batch_size)}
            cases.append(
                _CaseSpec(
                    name=f"uniform_{batch_size}x25",
                    steps=(_StepSpec(name=f"{batch_size}_slots_t25", slots=slots),),
                )
            )
    if enabled("staggered") and stream_slots >= 2:
        cases.append(
            _CaseSpec(
                name="staggered_two_slots",
                steps=(
                    _StepSpec(name="a_first", slots={0: 8}),
                    _StepSpec(name="a_b_together", slots={0: 8, 1: 8}),
                    _StepSpec(name="b_tail", slots={1: 8}, release_after=(0, 1)),
                ),
            )
        )
    if enabled("reuse"):
        cases.append(
            _CaseSpec(
                name="release_and_reuse_slot",
                steps=(
                    _StepSpec(name="first_request", slots={0: 13}, release_after=(0,)),
                    _StepSpec(
                        name="second_request_same_slot",
                        slots={0: 13},
                        release_after=(0,),
                    ),
                ),
            )
        )
    if enabled("long"):
        cases.append(
            _CaseSpec(
                name="single_long_context_10x100",
                steps=tuple(
                    _StepSpec(name=f"chunk_{index:02d}", slots={0: 100})
                    for index in range(10)
                ),
            )
        )
    if enabled("offline"):
        cases.append(_CaseSpec(name="offline_lane_while_session_live", steps=()))

    unknown = (
        selected - {"boundary", "multi", "staggered", "reuse", "offline", "long"}
        if selected
        else set()
    )
    if unknown:
        raise ValueError(f"unknown case selector(s): {sorted(unknown)}")
    return cases


def _decode_step(
    session: _CodecStreamSession,
    recorder: _StreamingTraceRecorder,
    *,
    case_name: str,
    step_index: int,
    step: _StepSpec,
    slot_map: dict[int, int],
    n_vq: int,
    vocab_size: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    plan = {
        slot_map[logical_slot]: _make_codes(
            n_vq=n_vq,
            frames=frames,
            vocab_size=vocab_size,
            generator=generator,
        )
        for logical_slot, frames in step.slots.items()
    }
    before_state = recorder.state_sample()
    start = time.perf_counter()
    with recorder.step(case_name, step_index, step.name):
        decoded = session.step(plan)
    elapsed_s = time.perf_counter() - start
    after_state = recorder.state_sample()
    return {
        "name": step.name,
        "logical_slots": {
            str(logical_slot): int(frames)
            for logical_slot, frames in step.slots.items()
        },
        "actual_slots": {
            str(logical_slot): int(slot_map[logical_slot])
            for logical_slot in step.slots
        },
        "release_after": list(step.release_after),
        "elapsed_s": elapsed_s,
        "outputs": {
            str(slot): _tensor_summary(audio) for slot, audio in sorted(decoded.items())
        },
        "state_before": before_state,
        "state_after": after_state,
    }


def _run_offline_lane_case(
    session: _CodecStreamSession,
    recorder: _StreamingTraceRecorder,
    *,
    n_vq: int,
    vocab_size: int,
    generator: torch.Generator,
    max_step_frames: int,
) -> dict[str, Any]:
    stream_slot = session.acquire()
    if stream_slot is None:
        raise RuntimeError("failed to acquire stream slot for offline-lane trace")
    try:
        stream_warmup = _StepSpec(name="stream_slot_warmup", slots={0: 25})
        warmup = _decode_step(
            session,
            recorder,
            case_name="offline_lane_while_session_live",
            step_index=0,
            step=stream_warmup,
            slot_map={0: stream_slot},
            n_vq=n_vq,
            vocab_size=vocab_size,
            generator=generator,
        )
        offline_codes = [
            _make_codes(
                n_vq=n_vq, frames=125, vocab_size=vocab_size, generator=generator
            ),
            _make_codes(
                n_vq=n_vq, frames=251, vocab_size=vocab_size, generator=generator
            ),
        ]
        before_state = recorder.state_sample()
        start = time.perf_counter()
        with recorder.step("offline_lane_while_session_live", 1, "decode_offline"):
            wavs = session.decode_offline(
                offline_codes, max_step_frames=max_step_frames
            )
        elapsed_s = time.perf_counter() - start
        after_state = recorder.state_sample()
        return {
            "name": "offline_lane_while_session_live",
            "steps": [
                warmup,
                {
                    "name": "decode_offline",
                    "input_frames": [int(codes.shape[1]) for codes in offline_codes],
                    "elapsed_s": elapsed_s,
                    "outputs": [_tensor_summary(wav) for wav in wavs],
                    "state_before": before_state,
                    "state_after": after_state,
                },
            ],
        }
    finally:
        session.release(stream_slot)


def _summarize_sdpa(records: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_errors = 0
    oracle_skipped = 0
    oracle_max_abs = 0.0
    oracle_compared = 0
    oracle_modes: Counter[str] = Counter()
    oracle_error_messages: Counter[str] = Counter()
    masked = 0
    mask_structure_records = 0
    mask_structure_skipped = 0
    mask_noncontiguous_rows = 0
    modules: dict[str, int] = {}
    for record in records:
        module = str(record.get("module"))
        modules[module] = modules.get(module, 0) + 1
        attn_mask = record.get("attn_mask")
        if attn_mask is not None:
            masked += 1
            if isinstance(attn_mask, dict):
                structure = attn_mask.get("structure")
                if isinstance(structure, dict):
                    mask_structure_records += 1
                    if "contiguity_scan_skipped" in structure:
                        mask_structure_skipped += 1
                    mask_noncontiguous_rows += int(
                        structure.get("noncontiguous_rows", 0)
                    )
        oracle = record.get("oracle")
        if isinstance(oracle, dict):
            if "error" in oracle:
                oracle_errors += 1
                oracle_error_messages[str(oracle["error"])] += 1
            if "skipped" in oracle:
                oracle_skipped += 1
            mode = oracle.get("mode")
            if isinstance(mode, str):
                oracle_modes[mode] += 1
            comparison = oracle.get("comparison")
            if isinstance(comparison, dict):
                oracle_compared += 1
                oracle_max_abs = max(
                    oracle_max_abs,
                    float(comparison.get("max_abs", 0.0)),
                )
    return {
        "count": len(records),
        "masked_calls": masked,
        "mask_structure_records": mask_structure_records,
        "mask_structure_skipped": mask_structure_skipped,
        "mask_noncontiguous_rows": mask_noncontiguous_rows,
        "oracle_errors": oracle_errors,
        "oracle_compared": oracle_compared,
        "oracle_error_messages": oracle_error_messages.most_common(5),
        "oracle_modes": dict(sorted(oracle_modes.items())),
        "oracle_skipped": oracle_skipped,
        "oracle_worst_max_abs": oracle_max_abs,
        "top_modules": sorted(modules.items(), key=lambda item: item[1], reverse=True)[
            :20
        ],
    }


def _summarize_sdpa_by_case(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("case")), []).append(record)
    return {case: _summarize_sdpa(items) for case, items in sorted(grouped.items())}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# MOSS-TTS Local Streaming Vocoder Trace",
        "",
        f"- model: `{report['model']}`",
        f"- codec_model: `{report['codec_model']}`",
        f"- device: `{report['device']}`",
        f"- torch: `{report['dependencies']['torch']}`",
        f"- cuda_sdp: `{report['cuda_sdp_settings']}`",
        f"- n_vq: `{report['n_vq']}`",
        f"- vocab_size: `{report['vocab_size']}`",
        f"- state_sample_root: `{report['state_sample_root']}`",
        f"- mask_structure_enabled: `{report['mask_structure_enabled']}`",
        (
            f"- attention implementations: "
            f"`{report['attention_implementation_summary']['attention_implementation_counts']}`"
        ),
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| cases | {len(report['cases'])} |",
        f"| sdpa calls | {report['sdpa_summary']['count']} |",
        f"| masked sdpa calls | {report['sdpa_summary']['masked_calls']} |",
        f"| mask structure records | {report['sdpa_summary']['mask_structure_records']} |",
        f"| mask noncontiguous rows | {report['sdpa_summary']['mask_noncontiguous_rows']} |",
        f"| oracle compared | {report['sdpa_summary']['oracle_compared']} |",
        f"| oracle skipped | {report['sdpa_summary']['oracle_skipped']} |",
        f"| oracle errors | {report['sdpa_summary']['oracle_errors']} |",
        f"| oracle worst max_abs | {report['sdpa_summary']['oracle_worst_max_abs']} |",
        f"| oracle modes | `{report['sdpa_summary']['oracle_modes']}` |",
        "",
        "## Cases",
        "",
        "| case | steps | elapsed s | output tensors |",
        "| --- | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        elapsed = sum(float(step.get("elapsed_s", 0.0)) for step in case["steps"])
        outputs = 0
        for step in case["steps"]:
            step_outputs = step.get("outputs", {})
            outputs += (
                len(step_outputs)
                if isinstance(step_outputs, dict)
                else len(step_outputs)
            )
        lines.append(
            f"| `{case['name']}` | {len(case['steps'])} | "
            f"{elapsed:.6f} | {outputs} |"
        )
    lines.extend(
        [
            "",
            "## SDPA By Case",
            "",
            "| case | calls | masked | mask structures | noncontig rows | oracle compared | oracle skipped | oracle errors | worst oracle max_abs |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case, summary in report["sdpa_summary_by_case"].items():
        lines.append(
            f"| `{case}` | {summary['count']} | {summary['masked_calls']} | "
            f"{summary['mask_structure_records']} | "
            f"{summary['mask_noncontiguous_rows']} | "
            f"{summary['oracle_compared']} | "
            f"{summary['oracle_skipped']} | {summary['oracle_errors']} | "
            f"{summary['oracle_worst_max_abs']} |"
        )
    lines.extend(
        ["", "## Top SDPA Modules", "", "| module | calls |", "| --- | ---: |"]
    )
    for module, count in report["sdpa_summary"]["top_modules"]:
        lines.append(f"| `{module}` | {count} |")
    if report["sdpa_summary"]["oracle_error_messages"]:
        lines.extend(
            [
                "",
                "## Top Oracle Errors",
                "",
                "| count | error |",
                "| ---: | --- |",
            ]
        )
        for error, count in report["sdpa_summary"]["oracle_error_messages"]:
            error_line = str(error).replace("\n", "<br>")
            lines.append(f"| {count} | `{error_line}` |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This harness traces the real persistent streaming codec session, not the non-streaming packed decoder.",
            "- `oracle_skipped` usually means the SDPA call used an attention mask that cannot be represented by the simple varlen FlashAttention oracle.",
            "- `mask noncontiguous rows` is only populated when `--mask-structure` is enabled and the mask is small enough for a full contiguity scan.",
            "- Large or masked SDPA records should be inspected in the JSON before deciding on `flash_attn_with_kvcache` vs packed-varlen streaming.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.disable_cudnn_sdp:
        _disable_cudnn_sdp()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = _load_moss_tts_local_processor(args.model)
    codec_model = args.codec_model or _resolve_audio_tokenizer_model_path(
        processor, None
    )
    audio_tokenizer = load_moss_tts_local_audio_tokenizer(
        codec_model, device=args.device
    )
    codec = audio_tokenizer.model
    n_vq = _num_codebooks(processor)
    vocab_size = _audio_vocab_size(processor)
    selected = set(args.case) if args.case else None
    cases = _build_cases(args.stream_slots, selected)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))

    report: dict[str, Any] = {
        "schema": "moss_tts_local_streaming_vocoder_trace_v1",
        "model": args.model,
        "codec_model": codec_model,
        "device": args.device,
        "dependencies": _dependency_summary(args.device),
        "cuda_sdp_settings": _cuda_sdp_settings(),
        "attention_implementation_summary": _attention_implementation_summary(codec),
        "n_vq": n_vq,
        "vocab_size": vocab_size,
        "stream_slots": int(args.stream_slots),
        "offline_slots": int(args.offline_slots),
        "max_step_frames": int(args.max_step_frames),
        "state_sample_root": args.state_sample_root,
        "mask_structure_enabled": bool(args.mask_structure),
        "cases": [],
    }

    session = _CodecStreamSession(
        codec,
        stream_slots=args.stream_slots,
        offline_slots=args.offline_slots,
        n_vq=n_vq,
    )
    try:
        with _StreamingTraceRecorder(
            codec,
            capture_oracle=not args.disable_oracle,
            state_sample_limit=args.state_sample_limit,
            state_sample_root=args.state_sample_root,
            capture_mask_structure=bool(args.mask_structure),
        ) as recorder:
            report["streaming_state_inventory"] = recorder.state_inventory()
            for case in cases:
                if case.name == "offline_lane_while_session_live":
                    report["cases"].append(
                        _run_offline_lane_case(
                            session,
                            recorder,
                            n_vq=n_vq,
                            vocab_size=vocab_size,
                            generator=generator,
                            max_step_frames=args.max_step_frames,
                        )
                    )
                    continue
                case_steps = []
                slot_map: dict[int, int] = {}
                for step_index, step in enumerate(case.steps):
                    for logical_slot in step.slots:
                        if logical_slot in slot_map:
                            continue
                        actual_slot = session.acquire()
                        if actual_slot is None:
                            raise RuntimeError(
                                f"failed to acquire stream slot for "
                                f"{case.name}:{step.name}"
                            )
                        slot_map[logical_slot] = actual_slot
                    case_steps.append(
                        _decode_step(
                            session,
                            recorder,
                            case_name=case.name,
                            step_index=step_index,
                            step=step,
                            slot_map=slot_map,
                            n_vq=n_vq,
                            vocab_size=vocab_size,
                            generator=generator,
                        )
                    )
                    for logical_slot in step.release_after:
                        actual_slot = slot_map.pop(logical_slot)
                        session.release(actual_slot)
                for actual_slot in slot_map.values():
                    session.release(actual_slot)
                report["cases"].append({"name": case.name, "steps": case_steps})
            report["module_records"] = recorder.module_records
            report["sdpa_records"] = recorder.sdpa_records
            report["sdpa_summary"] = _summarize_sdpa(recorder.sdpa_records)
            report["sdpa_summary_by_case"] = _summarize_sdpa_by_case(
                recorder.sdpa_records
            )
    finally:
        session.close()

    json_path = output_dir / "streaming_vocoder_trace.json"
    md_path = output_dir / "streaming_vocoder_trace.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    _write_markdown(report, md_path)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
