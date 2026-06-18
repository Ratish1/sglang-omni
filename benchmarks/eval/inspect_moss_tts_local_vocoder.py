# SPDX-License-Identifier: Apache-2.0
"""Inspect and parity-probe the MOSS-TTS Local vocoder decoder.

This is a development utility for the SGLang-backed vocoder work. It loads the
real MOSS-TTS Local processor, records the staged decoder topology, and compares
the current processor decode path with the scheduler's offline streaming-session
decode path.

Example:

    python -m benchmarks.eval.inspect_moss_tts_local_vocoder \
        --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
        --device cuda:0 \
        --output-dir /data/moss_vocoder_introspect \
        --probe 1x25 --probe 1x100 --probe 1x300 \
        --probe 8x100 --probe 8x300
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.moss_tts_local.stages import _load_moss_tts_local_processor
from sglang_omni.models.moss_tts_local.streaming_vocoder import _CodecStreamSession
from sglang_omni.models.moss_tts_local.vocoder_backend import (
    MossTTSLocalVocoderBackend,
    is_experimental_moss_tts_local_vocoder_backend,
    moss_tts_local_vocoder_backend_choices,
    parse_moss_tts_local_vocoder_backend,
)
from sglang_omni.models.moss_tts_local.vocoder_decoder import (
    build_moss_tts_local_vocoder_decoder,
    use_moss_tts_local_vocoder_decoder,
)
from sglang_omni.models.moss_tts_local.vocoder_introspection import (
    summarize_moss_tts_local_vocoder,
)
from sglang_omni.models.moss_tts_local.vocoder_sglang_patch import (
    get_moss_tts_local_sglang_vocoder_patch_info,
    install_moss_tts_local_sglang_vocoder_patch,
    uninstall_moss_tts_local_sglang_vocoder_patch,
)

logger = logging.getLogger(__name__)

_MOSS_ATTENTION_CLASS_NAME = "MossAudioTokenizerMultiheadAttention"


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


def _cuda_runtime_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        summary.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_device_capability": (
                    list(properties.major_minor)
                    if hasattr(properties, "major_minor")
                    else [properties.major, properties.minor]
                ),
                "cuda_device_total_memory": int(properties.total_memory),
            }
        )
    return summary


def _module_import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "available": True,
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
    }


def _git_commit_for_path(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path).resolve()
    directory = candidate if candidate.is_dir() else candidate.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _dependency_status() -> dict[str, Any]:
    flash_attn_status = _module_import_status("flash_attn")
    try:
        from flash_attn import flash_attn_varlen_func

        flash_attn_status["flash_attn_varlen_func"] = {
            "available": True,
            "signature": str(inspect.signature(flash_attn_varlen_func)),
        }
    except Exception as exc:
        flash_attn_status["flash_attn_varlen_func"] = {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    sglang_status = _module_import_status("sglang")
    sglang_flash_status: dict[str, Any]
    try:
        from sglang.jit_kernel import flash_attention as sglang_flash_attention

        sglang_flash_status = {
            "available": True,
            "file": getattr(sglang_flash_attention, "__file__", None),
            "flash_attn_varlen_signature": str(
                inspect.signature(sglang_flash_attention.flash_attn_varlen_func)
            ),
        }
    except Exception as exc:
        sglang_flash_status = {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    sglang_status["git_commit"] = _git_commit_for_path(sglang_status.get("file"))
    return {
        "flash_attn": flash_attn_status,
        "sglang": sglang_status,
        "sglang_flash_attention": sglang_flash_status,
    }


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _summarize_config_object(owner: Any) -> dict[str, Any]:
    if owner is None:
        return {}
    summary: dict[str, Any] = {}
    for name in dir(owner):
        if name.startswith("_"):
            continue
        try:
            value = getattr(owner, name)
        except Exception:
            continue
        if callable(value):
            continue
        if _is_json_scalar(value):
            summary[name] = value
        elif isinstance(value, (list, tuple)) and all(
            _is_json_scalar(item) for item in value
        ):
            summary[name] = list(value)
        elif isinstance(value, dict) and all(
            isinstance(key, str) and _is_json_scalar(item)
            for key, item in value.items()
        ):
            summary[name] = dict(value)
    return summary


def _summarize_config_source(label: str, owner: Any) -> dict[str, Any]:
    return {
        "label": label,
        "present": owner is not None,
        "class": (
            f"{owner.__class__.__module__}.{owner.__class__.__name__}"
            if owner is not None
            else None
        ),
        "values": _summarize_config_object(owner),
    }


def _config_sources(processor: Any) -> list[dict[str, Any]]:
    codec = getattr(processor, "audio_tokenizer", None)
    return [
        _summarize_config_source(
            "processor.model_config", getattr(processor, "model_config", None)
        ),
        _summarize_config_source(
            "processor.config", getattr(processor, "config", None)
        ),
        _summarize_config_source(
            "audio_tokenizer.config", getattr(codec, "config", None)
        ),
        _summarize_config_source(
            "audio_tokenizer.model_config", getattr(codec, "model_config", None)
        ),
    ]


def _resolve_sample_rate(processor: Any) -> int:
    for owner in (
        getattr(processor, "model_config", None),
        getattr(processor, "config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "model_config", None),
    ):
        if owner is None:
            continue
        value = getattr(owner, "sampling_rate", None)
        if value:
            return int(value)
    return 48000


def _parameter_dtype_summary(module: Any) -> dict[str, int]:
    if module is None or not hasattr(module, "parameters"):
        return {}
    counts: dict[str, int] = {}
    for parameter in module.parameters():
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + int(parameter.numel())
    return counts


def _model_config_summary(processor: Any) -> dict[str, Any]:
    codec = getattr(processor, "audio_tokenizer", None)
    return {
        "processor_class": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
        "config_sources": _config_sources(processor),
        "resolved_values": {
            "n_vq": _num_codebooks(processor),
            "audio_vocab_size": _audio_vocab_size(processor),
            "sampling_rate": _resolve_sample_rate(processor),
        },
        "processor_config": _summarize_config_object(
            getattr(processor, "model_config", None)
        ),
        "codec_class": (
            f"{codec.__class__.__module__}.{codec.__class__.__name__}"
            if codec is not None
            else None
        ),
        "codec_config": _summarize_config_object(getattr(codec, "config", None)),
        "codec_parameter_dtypes": _parameter_dtype_summary(codec),
    }


def _attention_implementation_summary(processor: Any) -> dict[str, Any]:
    codec = getattr(processor, "audio_tokenizer", None)
    decoder = getattr(codec, "decoder", None)
    attention_modules: list[Any] = []
    if decoder is not None and callable(getattr(decoder, "modules", None)):
        attention_modules = [
            module
            for module in decoder.modules()
            if module.__class__.__name__ == _MOSS_ATTENTION_CLASS_NAME
        ]
    implementation_counts: dict[str, int] = {}
    module_counts: dict[str, int] = {}
    module_globals: dict[str, Any] = {}
    for attention_module in attention_modules:
        implementation = str(
            getattr(attention_module, "attention_implementation", "unknown")
        )
        implementation_counts[implementation] = (
            implementation_counts.get(implementation, 0) + 1
        )
        module_name = attention_module.__class__.__module__
        module_counts[module_name] = module_counts.get(module_name, 0) + 1
        if module_name not in module_globals:
            remote_module = sys.modules.get(module_name)
            module_globals[module_name] = {
                "file": getattr(remote_module, "__file__", None),
                "HAS_FLASH_ATTN": getattr(remote_module, "HAS_FLASH_ATTN", None),
                "flash_attn_varlen_func": str(
                    getattr(remote_module, "flash_attn_varlen_func", None)
                ),
            }
    return {
        "attention_module_count": len(attention_modules),
        "attention_implementation_counts": implementation_counts,
        "attention_python_module_counts": module_counts,
        "attention_module_globals": module_globals,
    }


def _disable_cudnn_sdp() -> None:
    cuda_backend = getattr(torch.backends, "cuda", None)
    fn = getattr(cuda_backend, "enable_cudnn_sdp", None)
    if not callable(fn):
        raise RuntimeError("this PyTorch build does not expose enable_cudnn_sdp")
    fn(False)


def _audio_vocab_size(processor: Any) -> int:
    for owner in (
        getattr(processor, "model_config", None),
        getattr(processor, "config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "model_config", None),
    ):
        if owner is None:
            continue
        for attr in ("audio_vocab_size", "vocab_size", "codebook_size"):
            value = getattr(owner, attr, None)
            if value:
                return int(value)
    return 1024


def _num_codebooks(processor: Any) -> int:
    for owner in (
        getattr(processor, "model_config", None),
        getattr(processor, "config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "config", None),
        getattr(getattr(processor, "audio_tokenizer", None), "model_config", None),
    ):
        if owner is None:
            continue
        for attr in ("n_vq", "num_codebooks"):
            value = getattr(owner, attr, None)
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


def _make_codes_for_frames(
    *,
    frames_list: list[int],
    n_vq: int,
    vocab_size: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randint(0, vocab_size, (frames, n_vq), generator=generator)
        for frames in frames_list
    ]


def _time_call(
    label: str,
    fn,
    *,
    iterations: int,
    device: torch.device,
) -> tuple[list[Any], dict[str, Any]]:
    seconds: list[float] = []
    outputs: list[Any] | None = None
    for _ in range(iterations):
        _synchronize(device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = fn()
        _synchronize(device)
        seconds.append(time.perf_counter() - start)
    assert outputs is not None
    return outputs, {
        "label": label,
        "iterations": iterations,
        "seconds": seconds,
        "mean_seconds": sum(seconds) / len(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
    }


def _assert_codec_not_streaming(processor: Any, label: str) -> None:
    codec = getattr(processor, "audio_tokenizer", None)
    streaming_state = getattr(codec, "_streaming_state", None)
    if streaming_state is not None:
        raise RuntimeError(f"{label} left codec streaming state active")


def _owned_decode_audio_codes(
    processor: Any,
    codes_list: list[torch.Tensor],
    owned_decoder: Any,
) -> list[torch.Tensor]:
    codec = processor.audio_tokenizer
    _assert_codec_not_streaming(processor, "owned decoder")
    with use_moss_tts_local_vocoder_decoder(codec, owned_decoder):
        outputs = [
            torch.as_tensor(wav).detach().to("cpu", torch.float32)
            for wav in processor.decode_audio_codes(codes_list)
        ]
    _assert_codec_not_streaming(processor, "owned decoder")
    return outputs


def _decode_audio_codes_cpu(
    processor: Any,
    codes_list: list[torch.Tensor],
) -> list[torch.Tensor]:
    _assert_codec_not_streaming(processor, "processor decode")
    outputs = [
        torch.as_tensor(wav).detach().to("cpu", torch.float32)
        for wav in processor.decode_audio_codes(codes_list)
    ]
    _assert_codec_not_streaming(processor, "processor decode")
    return outputs


def _patch_info_fields(prefix: str, patch_info: Any) -> dict[str, Any]:
    return {
        f"{prefix}_attention_modules": int(patch_info.attention_modules),
        f"{prefix}_python_modules": int(patch_info.python_modules),
        f"{prefix}_ref_count": int(patch_info.ref_count),
        f"{prefix}_invocation_count": int(patch_info.invocation_count),
        f"{prefix}_attention_implementations": dict(
            patch_info.attention_implementations
        ),
    }


def _time_sglang_patch_decode_audio_codes(
    processor: Any,
    codes_list: list[torch.Tensor],
    *,
    iterations: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    codec = processor.audio_tokenizer
    install_info = install_moss_tts_local_sglang_vocoder_patch(codec)
    try:
        outputs, timing = _time_call(
            "sglang_patch.decode_audio_codes",
            lambda: _decode_audio_codes_cpu(processor, codes_list),
            iterations=iterations,
            device=device,
        )
        active_info = get_moss_tts_local_sglang_vocoder_patch_info(codec)
    finally:
        restored_info = uninstall_moss_tts_local_sglang_vocoder_patch(codec)
    timing.update(_patch_info_fields("install", install_info))
    timing.update(_patch_info_fields("active", active_info))
    timing.update(_patch_info_fields("restored", restored_info))
    return outputs, timing


def _tensor_output_summary(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = torch.as_tensor(tensor).detach().to("cpu", torch.float32)
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "device": str(cpu.device),
        "mean_abs": float(cpu.abs().mean().item()) if cpu.numel() else 0.0,
        "max_abs": float(cpu.abs().max().item()) if cpu.numel() else 0.0,
    }


def _summarize_comparisons(
    reference: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> dict[str, Any]:
    comparisons = [_compare_tensors(ref, out) for ref, out in zip(reference, candidate)]
    return {
        "outputs": [
            {
                **_tensor_output_summary(out),
                "comparison": comparison,
            }
            for out, comparison in zip(candidate, comparisons)
        ],
        "max_abs_delta": max(
            (float(comp.get("max_abs_delta", 0.0)) for comp in comparisons),
            default=0.0,
        ),
        "mean_abs_delta_max": max(
            (float(comp.get("mean_abs_delta", 0.0)) for comp in comparisons),
            default=0.0,
        ),
    }


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_cpu = torch.as_tensor(a).detach().to("cpu", torch.float32)
    b_cpu = torch.as_tensor(b).detach().to("cpu", torch.float32)
    same_shape = tuple(a_cpu.shape) == tuple(b_cpu.shape)
    common = tuple(min(x, y) for x, y in zip(a_cpu.shape, b_cpu.shape))
    slices = tuple(slice(0, n) for n in common)
    if len(common) != a_cpu.ndim or len(common) != b_cpu.ndim:
        return {"same_shape": same_shape, "comparable": False}
    delta = a_cpu[slices] - b_cpu[slices]
    signal = a_cpu[slices]
    noise = delta
    signal_energy = float((signal * signal).sum().item())
    noise_energy = float((noise * noise).sum().item())
    if noise_energy == 0.0:
        snr_db: float | str = "inf"
    elif signal_energy == 0.0:
        snr_db = "-inf"
    else:
        snr_db = 10.0 * math.log10(signal_energy / noise_energy)
    return {
        "same_shape": same_shape,
        "comparable": True,
        "common_shape": list(common),
        "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "mean_abs_delta": float(delta.abs().mean().item()) if delta.numel() else 0.0,
        "snr_db": snr_db,
    }


def _stress_cases() -> list[tuple[str, list[int]]]:
    return [
        ("single_short", [25]),
        ("single_typical", [100]),
        ("single_long", [300]),
        ("mixed_8", [25, 50, 75, 100, 125, 150, 200, 300]),
        (
            "mixed_16",
            [25, 50, 75, 100, 125, 150, 200, 300]
            + [32, 64, 96, 128, 160, 224, 320, 400],
        ),
    ]


def _run_stress_case(
    processor: Any,
    *,
    name: str,
    frames_list: list[int],
    iterations: int,
    seed: int,
    device: torch.device,
    compare_owned_decoder: bool,
    compare_sglang_patch: bool,
) -> dict[str, Any]:
    n_vq = _num_codebooks(processor)
    vocab_size = _audio_vocab_size(processor)
    codes_list = _make_codes_for_frames(
        frames_list=frames_list,
        n_vq=n_vq,
        vocab_size=vocab_size,
        seed=seed,
    )

    def processor_decode() -> list[torch.Tensor]:
        return [
            torch.as_tensor(wav).detach().to("cpu", torch.float32)
            for wav in processor.decode_audio_codes(codes_list)
        ]

    processor_outputs, processor_timing = _time_call(
        "processor.decode_audio_codes",
        processor_decode,
        iterations=iterations,
        device=device,
    )

    owned_timing: dict[str, Any] | None = None
    owned_outputs: list[torch.Tensor] | None = None
    if compare_owned_decoder:
        owned_decoder = build_moss_tts_local_vocoder_decoder(processor.audio_tokenizer)
        owned_outputs, owned_timing = _time_call(
            "owned_decoder.decode_audio_codes",
            lambda: _owned_decode_audio_codes(processor, codes_list, owned_decoder),
            iterations=iterations,
            device=device,
        )

    sglang_patch_timing: dict[str, Any] | None = None
    sglang_patch_outputs: list[torch.Tensor] | None = None
    if compare_sglang_patch:
        sglang_patch_outputs, sglang_patch_timing = (
            _time_sglang_patch_decode_audio_codes(
                processor,
                codes_list,
                iterations=iterations,
                device=device,
            )
        )

    result: dict[str, Any] = {
        "name": name,
        "batch_size": len(frames_list),
        "frames": frames_list,
        "total_frames": sum(frames_list),
        "max_frames": max(frames_list),
        "codebooks": n_vq,
        "vocab_size": vocab_size,
        "processor_decode": {
            **processor_timing,
            "outputs": [_tensor_output_summary(out) for out in processor_outputs],
        },
    }
    if owned_timing is not None and owned_outputs is not None:
        result["owned_decoder_decode"] = {
            **owned_timing,
            **_summarize_comparisons(processor_outputs, owned_outputs),
        }
    if sglang_patch_timing is not None and sglang_patch_outputs is not None:
        result["sglang_patch_decode"] = {
            **sglang_patch_timing,
            **_summarize_comparisons(processor_outputs, sglang_patch_outputs),
        }
    return result


def _run_probe(
    processor: Any,
    *,
    batch_size: int,
    frames: int,
    iterations: int,
    seed: int,
    max_step_frames: int,
    device: torch.device,
    compare_owned_decoder: bool,
    compare_sglang_patch: bool,
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

    def processor_decode() -> list[torch.Tensor]:
        return [
            torch.as_tensor(wav).detach().to("cpu", torch.float32)
            for wav in processor.decode_audio_codes(codes_list)
        ]

    processor_outputs, processor_timing = _time_call(
        "processor.decode_audio_codes",
        processor_decode,
        iterations=iterations,
        device=device,
    )

    codec = processor.audio_tokenizer
    owned_outputs: list[torch.Tensor] | None = None
    owned_timing: dict[str, Any] | None = None
    if compare_owned_decoder:
        owned_decoder = build_moss_tts_local_vocoder_decoder(codec)
        owned_outputs, owned_timing = _time_call(
            "owned_decoder.decode_audio_codes",
            lambda: _owned_decode_audio_codes(processor, codes_list, owned_decoder),
            iterations=iterations,
            device=device,
        )

    sglang_patch_outputs: list[torch.Tensor] | None = None
    sglang_patch_timing: dict[str, Any] | None = None
    if compare_sglang_patch:
        sglang_patch_outputs, sglang_patch_timing = (
            _time_sglang_patch_decode_audio_codes(
                processor,
                codes_list,
                iterations=iterations,
                device=device,
            )
        )

    channels_first = [codes.transpose(0, 1).contiguous() for codes in codes_list]

    session = _CodecStreamSession(
        codec,
        stream_slots=0,
        offline_slots=batch_size,
    )
    try:

        def session_decode() -> list[torch.Tensor]:
            return session.decode_offline(
                channels_first,
                max_step_frames=max_step_frames,
            )

        session_outputs, session_timing = _time_call(
            "session.decode_offline",
            session_decode,
            iterations=iterations,
            device=device,
        )
    finally:
        session.close()

    return {
        "batch_size": batch_size,
        "frames": frames,
        "codebooks": n_vq,
        "vocab_size": vocab_size,
        "processor_decode": {
            **processor_timing,
            "outputs": [_tensor_output_summary(out) for out in processor_outputs],
        },
        **(
            {
                "owned_decoder_decode": {
                    **owned_timing,
                    "outputs": [
                        {
                            **_tensor_output_summary(out),
                            "comparison": _compare_tensors(ref, out),
                        }
                        for ref, out in zip(processor_outputs, owned_outputs or [])
                    ],
                }
            }
            if owned_timing is not None
            else {}
        ),
        **(
            {
                "sglang_patch_decode": {
                    **sglang_patch_timing,
                    **_summarize_comparisons(
                        processor_outputs, sglang_patch_outputs or []
                    ),
                }
            }
            if sglang_patch_timing is not None
            else {}
        ),
        "session_offline_decode": {
            **session_timing,
            "outputs": [
                {
                    **_tensor_output_summary(out),
                    "comparison": _compare_tensors(ref, out),
                }
                for ref, out in zip(processor_outputs, session_outputs)
            ],
        },
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    decoder = report["introspection"]["decoder"]
    probes = report.get("decode_probes", [])
    lines = [
        "# MOSS-TTS Local Vocoder Introspection",
        "",
        f"- schema: `{report['schema']}`",
        f"- model: `{report['model_path']}`",
        f"- device: `{report['device']}`",
        f"- requested backend: `{report['requested_backend']}`",
        f"- backend experimental: `{report['requested_backend_experimental']}`",
        f"- cuda sdp: `{report['torch_backends']['cuda_sdp']}`",
        "",
        "## Decoder",
        "",
        f"- stage count: {decoder['stage_count']}",
        f"- transformer stages: {decoder['transformer_stage_count']}",
        f"- transformer layers: {decoder['transformer_layer_count']}",
        "",
        "| stage | type | input | hidden | output | layers | heads | head_dim | context |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in decoder["stages"]:
        lines.append(
            "| {stage_index} | {module_type} | {input_dimension} | {d_model} | "
            "{output_dimension} | {layers} | {heads} | {head_dim} | {context} |".format(
                stage_index=stage.get("stage_index"),
                module_type=stage.get("module_type"),
                input_dimension=stage.get("input_dimension", "-"),
                d_model=stage.get("d_model", "-"),
                output_dimension=stage.get("output_dimension", "-"),
                layers=stage.get("layers", "-"),
                heads=stage.get("heads", "-"),
                head_dim=stage.get("head_dim", "-"),
                context=stage.get("context", stage.get("patch_size", "-")),
            )
        )
    if probes:
        lines.extend(
            [
                "",
                "## Decode Probes",
                "",
                "| batch | frames | processor ms | owned ms | owned max delta | sglang patch ms | sglang patch max delta | session ms | session max delta |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for probe in probes:
        session_comparisons = [
            out.get("comparison", {})
            for out in probe["session_offline_decode"].get("outputs", [])
        ]
        session_max_delta = max(
            (float(comp.get("max_abs_delta", 0.0)) for comp in session_comparisons),
            default=0.0,
        )
        owned = probe.get("owned_decoder_decode")
        owned_ms = float("nan")
        owned_max_delta = float("nan")
        if owned is not None:
            owned_ms = owned["mean_seconds"] * 1000.0
            owned_max_delta = max(
                (
                    float(out.get("comparison", {}).get("max_abs_delta", 0.0))
                    for out in owned.get("outputs", [])
                ),
                default=0.0,
            )
        sglang_patch = probe.get("sglang_patch_decode")
        sglang_patch_ms = float("nan")
        sglang_patch_max_delta = float("nan")
        if sglang_patch is not None:
            sglang_patch_ms = sglang_patch["mean_seconds"] * 1000.0
            sglang_patch_max_delta = float(sglang_patch.get("max_abs_delta", 0.0))
        lines.append(
            "| {batch_size} | {frames} | {processor_ms:.3f} | "
            "{owned_ms:.3f} | {owned_max_delta:.6g} | "
            "{sglang_patch_ms:.3f} | {sglang_patch_max_delta:.6g} | "
            "{session_ms:.3f} | {session_max_delta:.6g} |".format(
                batch_size=probe["batch_size"],
                frames=probe["frames"],
                processor_ms=probe["processor_decode"]["mean_seconds"] * 1000.0,
                owned_ms=owned_ms,
                owned_max_delta=owned_max_delta,
                sglang_patch_ms=sglang_patch_ms,
                sglang_patch_max_delta=sglang_patch_max_delta,
                session_ms=probe["session_offline_decode"]["mean_seconds"] * 1000.0,
                session_max_delta=session_max_delta,
            )
        )
    stress_cases = report.get("stress_cases", [])
    if stress_cases:
        lines.extend(
            [
                "",
                "## Stress Cases",
                "",
                "| name | batch | total frames | max frames | processor ms | owned ms | owned max delta | sglang patch ms | sglang patch max delta |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for case in stress_cases:
        owned = case.get("owned_decoder_decode")
        owned_ms = float("nan")
        owned_max_delta = float("nan")
        if owned is not None:
            owned_ms = owned["mean_seconds"] * 1000.0
            owned_max_delta = float(owned.get("max_abs_delta", 0.0))
        sglang_patch = case.get("sglang_patch_decode")
        sglang_patch_ms = float("nan")
        sglang_patch_max_delta = float("nan")
        if sglang_patch is not None:
            sglang_patch_ms = sglang_patch["mean_seconds"] * 1000.0
            sglang_patch_max_delta = float(sglang_patch.get("max_abs_delta", 0.0))
        lines.append(
            "| {name} | {batch_size} | {total_frames} | {max_frames} | "
            "{processor_ms:.3f} | {owned_ms:.3f} | {owned_max_delta:.6g} | "
            "{sglang_patch_ms:.3f} | {sglang_patch_max_delta:.6g} |".format(
                name=case["name"],
                batch_size=case["batch_size"],
                total_frames=case["total_frames"],
                max_frames=case["max_frames"],
                processor_ms=case["processor_decode"]["mean_seconds"] * 1000.0,
                owned_ms=owned_ms,
                owned_max_delta=owned_max_delta,
                sglang_patch_ms=sglang_patch_ms,
                sglang_patch_max_delta=sglang_patch_max_delta,
            )
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
        "--backend",
        default=MossTTSLocalVocoderBackend.PROCESSOR.value,
        choices=moss_tts_local_vocoder_backend_choices(),
        help="vocoder backend to inspect; experimental backends are compared against processor",
    )
    parser.add_argument(
        "--probe",
        action="append",
        type=_parse_probe,
        default=[],
        help="probe shape in BxT form, e.g. 8x100; repeatable",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-step-frames", type=int, default=100)
    parser.add_argument(
        "--disable-cudnn-sdp",
        action="store_true",
        help="disable torch.backends.cuda cuDNN SDPA before loading/probing",
    )
    parser.add_argument(
        "--compare-owned-decoder",
        action="store_true",
        help="compare the experimental owned PyTorch vocoder decoder against the processor path",
    )
    parser.add_argument(
        "--compare-sglang-patch",
        action="store_true",
        help="compare processor decode after installing the SGLang FlashAttention patch against the unpatched processor path",
    )
    parser.add_argument(
        "--stress-suite",
        action="store_true",
        help="run deterministic mixed-length codec stress cases in addition to probes",
    )
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--no-markdown", action="store_true")
    parser.add_argument("--dump-env", action="store_true")
    parser.add_argument("--dump-model-config", action="store_true")
    parser.add_argument("--dump-attention-impl", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_arg_parser().parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    backend = parse_moss_tts_local_vocoder_backend(args.backend)
    if backend in {
        MossTTSLocalVocoderBackend.PROCESSOR_SDPA,
        MossTTSLocalVocoderBackend.PROCESSOR_FLASH2_UPSTREAM,
    }:
        raise SystemExit(
            f"--backend {backend.value!r} is reserved for the next SDPA/FA2 "
            "control phase; use --backend processor for the golden report"
        )
    compare_owned_decoder = (
        args.compare_owned_decoder
        or backend == MossTTSLocalVocoderBackend.OWNED_EXPERIMENTAL
    )
    compare_sglang_patch = (
        args.compare_sglang_patch
        or backend == MossTTSLocalVocoderBackend.SGLANG_PATCH_EXPERIMENTAL
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.disable_cudnn_sdp:
        logger.info("Disabling torch.backends.cuda cuDNN SDPA")
        _disable_cudnn_sdp()

    logger.info("Loading MOSS-TTS Local processor for %s", args.model)
    processor = _load_moss_tts_local_processor(args.model, device=args.device)
    device = _device_of_processor(processor, args.device)
    report: dict[str, Any] = {
        "schema": "moss_tts_local_vocoder_phase0_report_v1",
        "model_path": args.model,
        "device": str(device),
        "requested_backend": backend.value,
        "requested_backend_experimental": is_experimental_moss_tts_local_vocoder_backend(
            backend
        ),
        "torch_backends": {"cuda_sdp": _cuda_sdp_settings()},
        "introspection": summarize_moss_tts_local_vocoder(processor),
        "decode_probes": [],
        "stress_cases": [],
    }
    if args.dump_env:
        report["environment"] = {
            **_cuda_runtime_summary(device),
            "dependencies": _dependency_status(),
        }
    if args.dump_model_config:
        report["model_config_summary"] = _model_config_summary(processor)
    if args.dump_attention_impl:
        report["attention_implementation_summary"] = _attention_implementation_summary(
            processor
        )

    probes = args.probe or [(1, 25), (1, 100), (1, 300), (8, 100), (8, 300)]
    if not args.skip_probes:
        for index, (batch_size, frames) in enumerate(probes):
            logger.info("Running probe batch=%d frames=%d", batch_size, frames)
            report["decode_probes"].append(
                _run_probe(
                    processor,
                    batch_size=batch_size,
                    frames=frames,
                    iterations=args.iterations,
                    seed=args.seed + index,
                    max_step_frames=args.max_step_frames,
                    device=device,
                    compare_owned_decoder=compare_owned_decoder,
                    compare_sglang_patch=compare_sglang_patch,
                )
            )
    if args.stress_suite:
        for index, (name, frames_list) in enumerate(_stress_cases()):
            logger.info(
                "Running stress case %s batch=%d max_frames=%d total_frames=%d",
                name,
                len(frames_list),
                max(frames_list),
                sum(frames_list),
            )
            report["stress_cases"].append(
                _run_stress_case(
                    processor,
                    name=name,
                    frames_list=frames_list,
                    iterations=args.iterations,
                    seed=args.seed + 1000 + index,
                    device=device,
                    compare_owned_decoder=compare_owned_decoder,
                    compare_sglang_patch=compare_sglang_patch,
                )
            )

    json_path = output_dir / "moss_tts_local_vocoder_introspection.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %s", json_path)
    if not args.no_markdown:
        markdown_path = output_dir / "moss_tts_local_vocoder_introspection.md"
        _write_markdown(report, markdown_path)
        logger.info("Wrote %s", markdown_path)


if __name__ == "__main__":
    main()
