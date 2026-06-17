# SPDX-License-Identifier: Apache-2.0
"""Inspect the MOSS-TTS Local codec/vocoder before any SGLang adapter work.

This script is intentionally diagnostic. It loads the upstream remote-code
processor, inspects the audio tokenizer/decoder structure, and optionally runs
synthetic decode probes. It does not change serving behavior or install custom
kernels.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
import platform
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL_PATH = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"


@dataclass
class DecodeProbeConfig:
    enabled: bool
    session_offline: bool
    profile_torch: bool
    batch_size: int
    frames: list[int]
    codebooks: int | None
    vocab_size: int | None
    seed: int
    max_step_frames: int
    warmup: int
    iterations: int


@dataclass
class IntrospectionConfig:
    model_path: str
    device: str
    output_dir: Path
    tree_max_depth: int
    tree_max_children: int
    state_max_keys: int
    source_context_lines: int
    probe: DecodeProbeConfig


@dataclass
class Report:
    schema: str = "moss_tts_local_codec_introspection_v1"
    created_at_unix_s: float = field(default_factory=time.time)
    environment: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    memory_snapshots: list[dict[str, Any]] = field(default_factory=list)
    processor: dict[str, Any] = field(default_factory=dict)
    audio_tokenizer: dict[str, Any] = field(default_factory=dict)
    codec_decoder: dict[str, Any] = field(default_factory=dict)
    architecture_hints: dict[str, Any] = field(default_factory=dict)
    state_dict: dict[str, Any] = field(default_factory=dict)
    methods: dict[str, Any] = field(default_factory=dict)
    decode_probes: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


def json_safe(value: Any, *, max_string: int = 2000) -> Any:
    """Convert common Python/torch/numpy values into JSON-safe data."""
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value), max_string=max_string)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if isinstance(value, str) and len(value) > max_string:
            return (
                value[:max_string] + f"...<truncated {len(value) - max_string} chars>"
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, dict):
        return {str(k): json_safe(v, max_string=max_string) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v, max_string=max_string) for v in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        out: dict[str, Any] = {
            "shape": [int(x) for x in getattr(value, "shape", ())],
            "dtype": str(getattr(value, "dtype", "")),
        }
        device = getattr(value, "device", None)
        if device is not None:
            out["device"] = str(device)
        return out
    if hasattr(value, "item"):
        try:
            return json_safe(value.item(), max_string=max_string)
        except Exception:
            pass
    return repr(value)


def error_record(stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
    }


def class_record(obj: Any) -> dict[str, Any]:
    cls = obj if inspect.isclass(obj) else type(obj)
    return {
        "class_name": cls.__name__,
        "qualname": getattr(cls, "__qualname__", cls.__name__),
        "module": getattr(cls, "__module__", None),
        "source_location": source_location(cls),
    }


def source_location(obj: Any) -> dict[str, Any]:
    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        _, start_line = inspect.getsourcelines(obj)
        return {"file": source_file, "line": start_line}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def signature_record(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {"present": False}
    record = {"present": True, "source_location": source_location(obj)}
    try:
        record["signature"] = str(inspect.signature(obj))
    except Exception as exc:
        record["signature_error"] = f"{type(exc).__name__}: {exc}"
    return record


def source_snippet(obj: Any, *, max_lines: int) -> dict[str, Any]:
    try:
        lines, start_line = inspect.getsourcelines(obj)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    clipped = lines[: max(0, max_lines)]
    return {
        "start_line": start_line,
        "line_count": len(lines),
        "truncated": len(lines) > len(clipped),
        "text": "".join(clipped),
    }


def tensor_summary(tensor: Any) -> dict[str, Any]:
    return {
        "shape": [int(x) for x in getattr(tensor, "shape", ())],
        "dtype": str(getattr(tensor, "dtype", "")),
        "device": str(getattr(tensor, "device", "")),
        "numel": int(tensor.numel()) if hasattr(tensor, "numel") else None,
    }


def group_state_dict_keys(
    items: list[tuple[str, dict[str, Any]]], *, max_examples: int = 8
) -> list[dict[str, Any]]:
    """Group tensor keys by layer index templates for readable JSON reports."""
    grouped: dict[str, dict[str, Any]] = {}
    layer_re = re.compile(
        r"(?P<prefix>(?:^|\.)(?:h|layers|blocks|decoder_layers|transformer_layers|"
        r"model_layers|block))\.(?P<index>\d+)(?P<suffix>\.|$)"
    )
    for key, summary in items:
        match = layer_re.search(key)
        if match:
            index = int(match.group("index"))
            start, end = match.span("index")
            template = f"{key[:start]}{{layer}}{key[end:]}"
        else:
            parts = key.split(".")
            template = ".".join(parts[:2]) + ".*" if len(parts) > 2 else key
            index = None
        entry = grouped.setdefault(
            template,
            {
                "template": template,
                "count": 0,
                "layer_indices": set(),
                "examples": [],
            },
        )
        entry["count"] += 1
        if index is not None:
            entry["layer_indices"].add(index)
        if len(entry["examples"]) < max_examples:
            entry["examples"].append({"key": key, **summary})

    result = []
    for entry in grouped.values():
        indices = sorted(entry["layer_indices"])
        result.append(
            {
                "template": entry["template"],
                "count": entry["count"],
                "layer_indices": indices,
                "layer_count": len(indices) if indices else None,
                "examples": entry["examples"],
            }
        )
    result.sort(key=lambda x: (-int(x.get("layer_count") or 0), x["template"]))
    return result


def to_wave_tensor(value: Any):
    import torch

    tensor = torch.as_tensor(value).detach().to("cpu", torch.float32).contiguous()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def waveform_stats(a: Any, b: Any | None = None) -> dict[str, Any]:
    """Return scalar waveform stats and optional aligned delta metrics."""
    detached = to_wave_tensor(a)
    stats = {
        "shape": [int(x) for x in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
    }
    stats["max_abs"] = float(detached.abs().max().item()) if detached.numel() else 0.0
    stats["mean_abs"] = float(detached.abs().mean().item()) if detached.numel() else 0.0
    if b is not None:
        b_detached = to_wave_tensor(b)
        comparison: dict[str, Any] = {
            "shape_b": [int(x) for x in b_detached.shape],
            "same_shape": list(detached.shape) == list(b_detached.shape),
            "comparable": detached.ndim == b_detached.ndim,
        }
        if detached.ndim != b_detached.ndim:
            comparison["reason"] = "rank mismatch"
            stats["comparison"] = comparison
            return stats
        common_shape = [
            min(int(left), int(right))
            for left, right in zip(detached.shape, b_detached.shape)
        ]
        comparison["common_shape"] = common_shape
        if all(dim > 0 for dim in common_shape):
            slices = tuple(slice(0, dim) for dim in common_shape)
            aa = detached[slices].float()
            bb = b_detached[slices].float()
            noise = aa - bb
            signal_power = float((aa * aa).mean().item())
            noise_power = float((noise * noise).mean().item())
            comparison.update(
                {
                    "max_abs_delta": float(noise.abs().max().item()),
                    "mean_abs_delta": float(noise.abs().mean().item()),
                    "snr_db": (
                        float("inf")
                        if noise_power == 0.0
                        else 10.0 * math.log10(max(signal_power, 1e-30) / noise_power)
                    ),
                }
            )
        stats["comparison"] = comparison
    return stats


def parse_int_list(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def public_config_attrs(obj: Any) -> dict[str, Any]:
    out = {}
    if obj is None:
        return out
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            out[name] = f"<error {type(exc).__name__}: {exc}>"
            continue
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            out[name] = json_safe(value)
    return out


def memory_snapshot(label: str, device: str) -> dict[str, Any]:
    record: dict[str, Any] = {"label": label, "time_unix_s": time.time()}
    try:
        import resource

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        multiplier = 1024 if platform.system() != "Darwin" else 1
        record["process_max_rss_bytes"] = int(ru_maxrss) * multiplier
    except Exception as exc:
        record["process_max_rss_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        record["process_rss_bytes"] = int(mem.rss)
        record["process_vms_bytes"] = int(mem.vms)
    except Exception as exc:
        record["process_memory_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torch

        record["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            index = torch.device(device).index
            if index is None:
                index = torch.cuda.current_device()
            record["cuda_device"] = int(index)
            record["cuda_device_name"] = torch.cuda.get_device_name(index)
            record["cuda_memory_allocated_bytes"] = int(
                torch.cuda.memory_allocated(index)
            )
            record["cuda_memory_reserved_bytes"] = int(
                torch.cuda.memory_reserved(index)
            )
            record["cuda_max_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(index)
            )
            free, total = torch.cuda.mem_get_info(index)
            record["cuda_mem_get_info_free_bytes"] = int(free)
            record["cuda_mem_get_info_total_bytes"] = int(total)
    except Exception as exc:
        record["torch_cuda_error"] = f"{type(exc).__name__}: {exc}"
    return record


def find_decoder_candidates(codec: Any) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    direct_attrs = (
        ("decoder", 0),
        ("audio_decoder", 1),
        ("codec_decoder", 1),
        ("decode_model", 2),
        ("model", 10),
    )
    order = 0
    for attr, priority in direct_attrs:
        value = getattr(codec, attr, None)
        if value is not None and id(value) not in seen:
            seen.add(id(value))
            candidates.append(
                {
                    "name": attr,
                    "module": value,
                    "reason": "direct_attr",
                    "priority": priority,
                    "order": order,
                }
            )
            order += 1
    if hasattr(codec, "named_modules"):
        for name, module in codec.named_modules():
            lowered = name.lower()
            if name and ("decoder" in lowered or lowered.endswith("decode")):
                if id(module) not in seen:
                    seen.add(id(module))
                    base = lowered.rsplit(".", 1)[-1]
                    priority = 0 if base == "decoder" else 3
                    candidates.append(
                        {
                            "name": name,
                            "module": module,
                            "reason": "named_modules",
                            "priority": priority,
                            "order": order,
                        }
                    )
                    order += 1
    candidates.sort(key=lambda item: (item["priority"], item["order"], item["name"]))
    return candidates


def direct_param_count(module: Any) -> int:
    total = 0
    try:
        for param in module.parameters(recurse=False):
            total += int(param.numel())
    except Exception:
        return 0
    return total


def module_tree(module: Any, *, max_depth: int, max_children: int) -> dict[str, Any]:
    def walk(current: Any, name: str, depth: int) -> dict[str, Any]:
        node = {
            "name": name,
            **class_record(current),
            "direct_parameter_count": direct_param_count(current),
        }
        if depth >= max_depth or not hasattr(current, "named_children"):
            return node
        children = []
        for index, (child_name, child) in enumerate(current.named_children()):
            if index >= max_children:
                children.append({"name": "<truncated>", "remaining_children": True})
                break
            children.append(walk(child, child_name, depth + 1))
        if children:
            node["children"] = children
        return node

    return walk(module, "", 0)


def infer_architecture_hints(codec: Any, decoder: Any | None) -> dict[str, Any]:
    root = decoder if decoder is not None else codec
    hints: dict[str, Any] = {
        "config_attrs": {},
        "module_scans": {},
        "source_keyword_hits": {},
    }
    for label, obj in (
        ("codec.config", getattr(codec, "config", None)),
        ("codec.model_config", getattr(codec, "model_config", None)),
        ("decoder.config", getattr(root, "config", None)),
    ):
        attrs = public_config_attrs(obj)
        if attrs:
            hints["config_attrs"][label] = attrs

    named_modules = list(root.named_modules()) if hasattr(root, "named_modules") else []
    class_names = [(name, type(module).__name__) for name, module in named_modules]
    hints["module_scans"]["norm_types"] = sorted(
        {cls for _, cls in class_names if "norm" in cls.lower()}
    )
    hints["module_scans"]["activation_types"] = sorted(
        {
            cls
            for _, cls in class_names
            if any(token in cls.lower() for token in ("gelu", "silu", "relu", "swish"))
        }
    )
    hints["module_scans"]["rope_like_modules"] = [
        {"name": name, "class_name": cls}
        for name, cls in class_names
        if any(token in (name + cls).lower() for token in ("rope", "rotary", "alibi"))
    ]

    layer_candidates = []
    for name, module in named_modules:
        cls = type(module).__name__
        if cls in {"ModuleList", "Sequential"} or any(
            token in name.lower() for token in ("layers", "blocks", ".h")
        ):
            try:
                length = len(module)
            except Exception:
                continue
            if length > 0:
                layer_candidates.append(
                    {"name": name, "class_name": cls, "length": int(length)}
                )
    layer_candidates.sort(key=lambda item: (-item["length"], item["name"]))
    hints["module_scans"]["layer_candidates"] = layer_candidates[:20]

    linear_shapes = []
    for name, module in named_modules:
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "shape"):
            continue
        shape = [int(x) for x in weight.shape]
        if len(shape) == 2:
            linear_shapes.append(
                {
                    "name": name,
                    "class_name": type(module).__name__,
                    "weight_shape": shape,
                }
            )
    hints["module_scans"]["linear_weight_shapes"] = linear_shapes[:80]

    keyword_hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keywords = ("causal", "mask", "attn_mask", "tril", "rotary", "rope", "gelu")
    for name, module in named_modules[:400]:
        try:
            text = inspect.getsource(type(module))
        except Exception:
            continue
        lowered = text.lower()
        for keyword in keywords:
            if keyword in lowered and len(keyword_hits[keyword]) < 20:
                keyword_hits[keyword].append(
                    {"name": name, "class_name": type(module).__name__}
                )
    hints["source_keyword_hits"] = dict(keyword_hits)
    hints["likely"] = summarize_likely_architecture(hints)
    return hints


def summarize_likely_architecture(hints: dict[str, Any]) -> dict[str, Any]:
    attrs = {}
    for values in hints.get("config_attrs", {}).values():
        attrs.update(values)

    def pick(*names: str) -> Any:
        for name in names:
            if name in attrs:
                return attrs[name]
        return None

    likely = {
        "layer_count": pick("num_hidden_layers", "n_layer", "num_layers"),
        "hidden_size": pick("hidden_size", "n_embd", "d_model", "dim"),
        "head_count": pick("num_attention_heads", "n_head", "num_heads"),
        "head_dim": pick("head_dim", "attention_head_dim"),
        "ffn_size": pick("intermediate_size", "n_inner", "ffn_dim", "mlp_dim"),
        "norm_type": None,
        "activation": pick("hidden_act", "activation", "activation_function"),
        "rope_style": None,
        "causal_or_mask_hints": {},
    }
    layers = hints.get("module_scans", {}).get("layer_candidates", [])
    if likely["layer_count"] is None and layers:
        likely["layer_count"] = layers[0]["length"]
    if likely["norm_type"] is None:
        norms = hints.get("module_scans", {}).get("norm_types", [])
        likely["norm_type"] = norms
    if likely["activation"] is None:
        likely["activation"] = hints.get("module_scans", {}).get("activation_types", [])
    rope_modules = hints.get("module_scans", {}).get("rope_like_modules", [])
    source_hits = hints.get("source_keyword_hits", {})
    likely["rope_style"] = {
        "modules": rope_modules[:20],
        "source_hits": {
            key: source_hits.get(key, []) for key in ("rope", "rotary", "alibi")
        },
    }
    likely["causal_or_mask_hints"] = {
        key: source_hits.get(key, [])
        for key in ("causal", "mask", "attn_mask", "tril")
        if source_hits.get(key)
    }
    return likely


def inspect_state_dict(module: Any, *, max_keys: int) -> dict[str, Any]:
    state = module.state_dict()
    items = [(key, tensor_summary(value)) for key, value in state.items()]
    total_numel = sum(int(summary.get("numel") or 0) for _, summary in items)
    return {
        "tensor_count": len(items),
        "total_numel": total_numel,
        "total_parameter_bytes_estimate": sum(
            int(summary.get("numel") or 0)
            * {
                "torch.float16": 2,
                "torch.bfloat16": 2,
                "torch.float32": 4,
                "torch.float64": 8,
                "torch.int64": 8,
                "torch.int32": 4,
                "torch.int16": 2,
                "torch.int8": 1,
                "torch.uint8": 1,
                "torch.bool": 1,
            }.get(str(summary.get("dtype")), 0)
            for _, summary in items
        ),
        "groups": group_state_dict_keys(items),
        "keys": [{"key": key, **summary} for key, summary in items[: max(0, max_keys)]],
        "keys_truncated": len(items) > max_keys,
    }


def synchronize_if_cuda(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def timed_call(label: str, fn: Any, *, device: str, iterations: int) -> dict[str, Any]:
    timings = []
    result = None
    for _ in range(iterations):
        synchronize_if_cuda(device)
        start = time.perf_counter()
        result = fn()
        synchronize_if_cuda(device)
        timings.append(time.perf_counter() - start)
    return {
        "label": label,
        "iterations": iterations,
        "seconds": timings,
        "mean_seconds": sum(timings) / len(timings) if timings else None,
        "result": result,
    }


def streaming_state_summary(module: Any, *, limit: int = 20) -> dict[str, Any]:
    if not hasattr(module, "named_modules"):
        return {"count": 0, "examples": []}
    examples = []
    total = 0
    for name, child in module.named_modules():
        state = getattr(child, "_streaming_state", None)
        if state is None:
            continue
        total += 1
        if len(examples) < limit:
            examples.append(
                {
                    "module": name or "<root>",
                    "module_class": type(child).__name__,
                    "state_class": type(state).__name__,
                    "state_attrs": public_config_attrs(state),
                }
            )
    return {"count": total, "examples": examples}


def value_shape_record(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return tensor_summary(value)
    if hasattr(value, "audio") or hasattr(value, "audio_lengths"):
        record = {"type": type(value).__name__}
        for attr in ("audio", "audio_lengths"):
            item = getattr(value, attr, None)
            if item is not None:
                record[attr] = value_shape_record(item)
        return record
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def capture_decode_frame_calls(
    codec: Any,
    fn: Any,
    *,
    max_records: int = 32,
) -> dict[str, Any]:
    original = getattr(codec, "_decode_frame", None)
    if original is None:
        return {"ok": False, "reason": "codec has no _decode_frame"}

    records: list[dict[str, Any]] = []

    def wrapped(*args: Any, **kwargs: Any):
        should_record = len(records) < max_records
        record: dict[str, Any] | None = None
        if should_record:
            record = {
                "inputs": [value_shape_record(arg) for arg in args],
                "kwargs": {
                    key: value_shape_record(value) for key, value in kwargs.items()
                },
                "streaming_state_before": streaming_state_summary(codec, limit=8),
            }
        result = original(*args, **kwargs)
        if record is not None:
            record["output"] = value_shape_record(result)
            record["streaming_state_after"] = streaming_state_summary(codec, limit=8)
            records.append(record)
        return result

    try:
        setattr(codec, "_decode_frame", wrapped)
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"failed to wrap _decode_frame: {type(exc).__name__}: {exc}",
        }

    try:
        fn()
    except Exception as exc:
        return {
            "ok": False,
            "records": records,
            "error": error_record("_decode_frame_capture", exc),
        }
    finally:
        setattr(codec, "_decode_frame", original)

    return {
        "ok": True,
        "count": len(records),
        "records": records,
        "truncated": len(records) >= max_records,
    }


def profile_call(label: str, fn: Any, *, device: str, trace_path: Path) -> None:
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(device).startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    synchronize_if_cuda(device)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.profiler.record_function(label):
            fn()
    synchronize_if_cuda(device)
    prof.export_chrome_trace(str(trace_path))


def make_synthetic_codes(
    *,
    batch_size: int,
    frames: int,
    codebooks: int,
    vocab_size: int,
    seed: int,
) -> list[Any]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.randint(
        low=0,
        high=max(int(vocab_size), 1),
        size=(batch_size, frames, codebooks),
        generator=generator,
        dtype=torch.long,
    )
    return [row for row in rows]


def run_decode_probe(
    processor: Any,
    codec: Any,
    cfg: DecodeProbeConfig,
    *,
    device: str,
    output_dir: Path,
    inferred_codebooks: int,
    inferred_vocab_size: int,
) -> list[dict[str, Any]]:
    import torch

    probes = []
    codebooks = int(cfg.codebooks or inferred_codebooks)
    vocab_size = int(cfg.vocab_size or inferred_vocab_size)
    for frames in cfg.frames:
        entry: dict[str, Any] = {
            "batch_size": cfg.batch_size,
            "frames": int(frames),
            "codebooks": codebooks,
            "vocab_size": vocab_size,
        }
        codes_list = make_synthetic_codes(
            batch_size=cfg.batch_size,
            frames=int(frames),
            codebooks=codebooks,
            vocab_size=vocab_size,
            seed=cfg.seed + int(frames),
        )
        for _ in range(max(cfg.warmup, 0)):
            with torch.no_grad():
                processor.decode_audio_codes(codes_list)
        processor_wavs = None

        def processor_decode_once():
            with torch.no_grad():
                return processor.decode_audio_codes(codes_list)

        try:
            with torch.no_grad():
                timed = timed_call(
                    "processor.decode_audio_codes",
                    processor_decode_once,
                    device=device,
                    iterations=max(cfg.iterations, 1),
                )
            processor_wavs = timed.pop("result")
            entry["processor_decode"] = timed
            entry["processor_decode"]["outputs"] = [
                waveform_stats(wav) for wav in processor_wavs
            ]
            entry["processor_decode"]["decode_frame_calls"] = (
                capture_decode_frame_calls(
                    codec,
                    processor_decode_once,
                )
            )
            if cfg.profile_torch:
                trace_path = (
                    output_dir
                    / f"processor_decode_b{cfg.batch_size}_f{int(frames)}.trace.json"
                )
                with torch.no_grad():
                    profile_call(
                        "moss_probe.processor_decode_audio_codes",
                        processor_decode_once,
                        device=device,
                        trace_path=trace_path,
                    )
                entry["processor_decode"]["torch_trace"] = str(trace_path)
        except Exception as exc:
            entry["processor_decode_error"] = error_record(
                "processor.decode_audio_codes", exc
            )

        if cfg.session_offline:
            try:
                from sglang_omni.models.moss_tts_local.streaming_vocoder import (
                    _CodecStreamSession,
                )

                channels_first = [
                    row[:, :codebooks].transpose(0, 1).contiguous()
                    for row in codes_list
                ]

                def decode_with_session(active_session):
                    with torch.no_grad():
                        return active_session.decode_offline(
                            channels_first,
                            max_step_frames=cfg.max_step_frames,
                        )

                with torch.no_grad():
                    session = _CodecStreamSession(
                        codec,
                        stream_slots=0,
                        offline_slots=max(cfg.batch_size, 1),
                    )
                    try:
                        timed = timed_call(
                            "session.decode_offline",
                            lambda: decode_with_session(session),
                            device=device,
                            iterations=max(cfg.iterations, 1),
                        )
                    finally:
                        session.close()
                session_wavs = timed.pop("result")
                entry["session_offline_decode"] = timed
                outputs = []
                for index, wav in enumerate(session_wavs):
                    compare = processor_wavs[index] if processor_wavs else None
                    outputs.append(waveform_stats(wav, compare))
                entry["session_offline_decode"]["outputs"] = outputs
                session = _CodecStreamSession(
                    codec,
                    stream_slots=0,
                    offline_slots=max(cfg.batch_size, 1),
                )
                try:
                    entry["session_offline_decode"]["decode_frame_calls"] = (
                        capture_decode_frame_calls(
                            codec,
                            lambda: decode_with_session(session),
                        )
                    )
                finally:
                    session.close()
                if cfg.profile_torch:
                    trace_path = (
                        output_dir
                        / f"session_offline_decode_b{cfg.batch_size}_f{int(frames)}.trace.json"
                    )
                    session = _CodecStreamSession(
                        codec,
                        stream_slots=0,
                        offline_slots=max(cfg.batch_size, 1),
                    )
                    try:
                        with torch.no_grad():
                            profile_call(
                                "moss_probe.session_decode_offline",
                                lambda: decode_with_session(session),
                                device=device,
                                trace_path=trace_path,
                            )
                    finally:
                        session.close()
                    entry["session_offline_decode"]["torch_trace"] = str(trace_path)
            except Exception as exc:
                entry["session_offline_decode_error"] = error_record(
                    "session.decode_offline", exc
                )
        probes.append(entry)
    return probes


def build_report(cfg: IntrospectionConfig) -> tuple[Report, int]:
    report = Report()
    report.environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }
    report.config = json_safe(cfg)
    report.memory_snapshots.append(memory_snapshot("before_load", cfg.device))

    try:
        from sglang_omni.models.moss_tts_local.stages import (
            _load_moss_tts_local_processor,
        )

        processor = _load_moss_tts_local_processor(cfg.model_path, device=cfg.device)
    except Exception as exc:
        report.errors.append(error_record("processor_load", exc))
        report.memory_snapshots.append(memory_snapshot("after_load_error", cfg.device))
        return report, 2

    report.memory_snapshots.append(memory_snapshot("after_processor_load", cfg.device))
    report.processor = class_record(processor)
    report.processor["model_config"] = public_config_attrs(
        getattr(processor, "model_config", None)
    )

    codec = getattr(processor, "audio_tokenizer", None)
    if codec is None:
        report.errors.append(
            {
                "stage": "audio_tokenizer_lookup",
                "type": "MissingAttribute",
                "message": "processor.audio_tokenizer is missing",
            }
        )
        return report, 3

    report.audio_tokenizer = class_record(codec)
    report.audio_tokenizer["config"] = public_config_attrs(
        getattr(codec, "config", None)
    )
    report.methods["_decode_frame"] = signature_record(
        getattr(codec, "_decode_frame", None)
    )
    report.methods["decode_audio_codes"] = signature_record(
        getattr(processor, "decode_audio_codes", None)
    )
    if cfg.source_context_lines > 0:
        decode_frame = getattr(codec, "_decode_frame", None)
        decode_codes = getattr(processor, "decode_audio_codes", None)
        report.methods["_decode_frame"]["source_snippet"] = (
            source_snippet(decode_frame, max_lines=cfg.source_context_lines)
            if decode_frame is not None
            else None
        )
        report.methods["decode_audio_codes"]["source_snippet"] = (
            source_snippet(decode_codes, max_lines=cfg.source_context_lines)
            if decode_codes is not None
            else None
        )

    candidates = find_decoder_candidates(codec)
    decoder = candidates[0]["module"] if candidates else codec
    report.codec_decoder = {
        "selected": {
            "name": candidates[0]["name"] if candidates else "<audio_tokenizer>",
            "reason": candidates[0]["reason"] if candidates else "fallback",
            **class_record(decoder),
        },
        "candidates": [
            {
                "name": item["name"],
                "reason": item["reason"],
                **class_record(item["module"]),
            }
            for item in candidates
        ],
        "module_tree": module_tree(
            decoder,
            max_depth=cfg.tree_max_depth,
            max_children=cfg.tree_max_children,
        ),
    }
    report.architecture_hints = infer_architecture_hints(codec, decoder)
    try:
        report.state_dict = inspect_state_dict(decoder, max_keys=cfg.state_max_keys)
    except Exception as exc:
        report.errors.append(error_record("state_dict_inspection", exc))

    if cfg.probe.enabled:
        model_config = getattr(processor, "model_config", None)
        inferred_codebooks = int(
            cfg.probe.codebooks
            or getattr(model_config, "n_vq", 0)
            or getattr(getattr(codec, "config", None), "n_vq", 0)
            or 12
        )
        inferred_vocab_size = int(
            cfg.probe.vocab_size
            or getattr(model_config, "audio_vocab_size", 0)
            or getattr(getattr(codec, "config", None), "audio_vocab_size", 0)
            or 1024
        )
        report.decode_probes = run_decode_probe(
            processor,
            codec,
            cfg.probe,
            device=cfg.device,
            output_dir=cfg.output_dir,
            inferred_codebooks=inferred_codebooks,
            inferred_vocab_size=inferred_vocab_size,
        )
        report.memory_snapshots.append(
            memory_snapshot("after_decode_probes", cfg.device)
        )

    return report, 0


def render_markdown(report: Report) -> str:
    data = json_safe(report)
    lines = [
        "# MOSS-TTS Local Codec Introspection",
        "",
        f"- Schema: `{data['schema']}`",
        f"- Model path: `{data.get('config', {}).get('model_path')}`",
        f"- Device: `{data.get('config', {}).get('device')}`",
        "",
        "## Processor",
        "",
        "- Class: "
        f"`{data.get('processor', {}).get('module')}."
        f"{data.get('processor', {}).get('qualname')}`",
        f"- Source: `{data.get('processor', {}).get('source_location')}`",
        "",
        "## Audio Tokenizer",
        "",
        "- Class: "
        f"`{data.get('audio_tokenizer', {}).get('module')}."
        f"{data.get('audio_tokenizer', {}).get('qualname')}`",
        f"- Source: `{data.get('audio_tokenizer', {}).get('source_location')}`",
        "",
        "## Methods",
        "",
    ]
    for name, method in data.get("methods", {}).items():
        lines.append(
            f"- `{name}`: `{method.get('signature')}` "
            f"at `{method.get('source_location')}`"
        )
    lines.extend(["", "## Likely Architecture", "", "```json"])
    lines.append(
        json.dumps(data.get("architecture_hints", {}).get("likely", {}), indent=2)
    )
    lines.extend(["```", "", "## Decoder", ""])
    selected = data.get("codec_decoder", {}).get("selected", {})
    lines.append(
        f"- Selected: `{selected.get('name')}` "
        f"(`{selected.get('module')}.{selected.get('qualname')}`)"
    )
    lines.append(f"- Source: `{selected.get('source_location')}`")
    lines.extend(["", "## State Dict Groups", ""])
    for group in data.get("state_dict", {}).get("groups", [])[:80]:
        layer_count = group.get("layer_count")
        suffix = f", layers={layer_count}" if layer_count else ""
        lines.append(f"- `{group['template']}`: count={group['count']}{suffix}")
    lines.extend(["", "## Memory Snapshots", ""])
    for snap in data.get("memory_snapshots", []):
        cuda_alloc = snap.get("cuda_memory_allocated_bytes")
        rss = snap.get("process_rss_bytes") or snap.get("process_max_rss_bytes")
        lines.append(f"- `{snap.get('label')}`: rss={rss}, cuda_allocated={cuda_alloc}")
    lines.extend(["", "## Decode Probes", ""])
    if data.get("decode_probes"):
        for probe in data["decode_probes"]:
            lines.append(
                f"- batch={probe.get('batch_size')} frames={probe.get('frames')} "
                f"codebooks={probe.get('codebooks')}"
            )
            for key in ("processor_decode", "session_offline_decode"):
                if key in probe:
                    lines.append(
                        f"  - {key}: mean_seconds={probe[key].get('mean_seconds')}"
                    )
                error_key = key + "_error"
                if error_key in probe:
                    err = probe[error_key]
                    lines.append(
                        f"  - {error_key}: {err.get('type')}: {err.get('message')}"
                    )
    else:
        lines.append("- Not run.")
    if data.get("errors"):
        lines.extend(["", "## Errors", ""])
        for err in data["errors"]:
            lines.append(
                f"- `{err.get('stage')}`: {err.get('type')}: {err.get('message')}"
            )
    lines.append("")
    return "\n".join(lines)


def write_reports(report: Report, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "moss_tts_local_codec_introspection.json"
    md_path = output_dir / "moss_tts_local_codec_introspection.md"
    json_path.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def parse_args(argv: list[str]) -> IntrospectionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("moss_tts_local_codec_introspection"),
    )
    parser.add_argument("--tree-max-depth", type=int, default=5)
    parser.add_argument("--tree-max-children", type=int, default=80)
    parser.add_argument("--state-max-keys", type=int, default=500)
    parser.add_argument("--source-context-lines", type=int, default=80)
    parser.add_argument("--probe-decode", action="store_true")
    parser.add_argument("--probe-session-offline", action="store_true")
    parser.add_argument("--profile-torch", action="store_true")
    parser.add_argument("--probe-batch-size", type=int, default=1)
    parser.add_argument("--probe-frames", type=parse_int_list, default=[25, 100])
    parser.add_argument("--probe-codebooks", type=int, default=None)
    parser.add_argument("--probe-vocab-size", type=int, default=None)
    parser.add_argument("--probe-seed", type=int, default=1234)
    parser.add_argument("--max-step-frames", type=int, default=100)
    parser.add_argument("--probe-warmup", type=int, default=1)
    parser.add_argument("--probe-iterations", type=int, default=3)
    args = parser.parse_args(argv)
    return IntrospectionConfig(
        model_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        tree_max_depth=args.tree_max_depth,
        tree_max_children=args.tree_max_children,
        state_max_keys=args.state_max_keys,
        source_context_lines=args.source_context_lines,
        probe=DecodeProbeConfig(
            enabled=bool(args.probe_decode),
            session_offline=bool(args.probe_session_offline),
            profile_torch=bool(args.profile_torch),
            batch_size=args.probe_batch_size,
            frames=args.probe_frames,
            codebooks=args.probe_codebooks,
            vocab_size=args.probe_vocab_size,
            seed=args.probe_seed,
            max_step_frames=args.max_step_frames,
            warmup=args.probe_warmup,
            iterations=args.probe_iterations,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    report, exit_code = build_report(cfg)
    json_path, md_path = write_reports(report, cfg.output_dir)
    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {md_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
