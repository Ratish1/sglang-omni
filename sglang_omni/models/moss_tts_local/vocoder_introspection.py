# SPDX-License-Identifier: Apache-2.0
"""Introspection helpers for the MOSS-TTS Local vocoder decoder."""

from __future__ import annotations

import inspect
import re
from collections import defaultdict
from typing import Any

import torch
from torch import nn

_LAYER_PARAM_RE = re.compile(r"^(\d+)\.transformer\.layers\.(\d+)\.(.+)$")


def source_location(obj: Any) -> dict[str, Any] | None:
    """Return a stable source location for Python-defined objects."""
    try:
        file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        _, line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    return {"file": file, "line": line}


def source_snippet(obj: Any, *, max_lines: int = 80) -> dict[str, Any] | None:
    """Return a bounded source snippet for audit/debug artifacts."""
    try:
        lines, start_line = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    truncated = len(lines) > max_lines
    if truncated:
        lines = lines[:max_lines]
    return {
        "start_line": start_line,
        "line_count": len(lines),
        "text": "".join(lines),
        "truncated": truncated,
    }


def class_summary(obj: Any) -> dict[str, Any]:
    cls = obj if isinstance(obj, type) else obj.__class__
    return {
        "class_name": cls.__name__,
        "module": cls.__module__,
        "qualname": cls.__qualname__,
        "source_location": source_location(cls),
    }


def tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
    }


def linear_summary(module: Any) -> dict[str, Any] | None:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None
    bias = getattr(module, "bias", None)
    return {
        "class_name": module.__class__.__name__,
        "weight": tensor_summary(weight),
        "bias": tensor_summary(bias) if isinstance(bias, torch.Tensor) else None,
        "in_features": int(getattr(module, "in_features", weight.shape[1])),
        "out_features": int(getattr(module, "out_features", weight.shape[0])),
    }


def norm_summary(module: Any) -> dict[str, Any] | None:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None
    bias = getattr(module, "bias", None)
    return {
        "class_name": module.__class__.__name__,
        "eps": getattr(module, "eps", None),
        "weight": tensor_summary(weight),
        "bias": tensor_summary(bias) if isinstance(bias, torch.Tensor) else None,
    }


def method_summary(
    obj: Any, method_name: str, *, max_lines: int = 80
) -> dict[str, Any]:
    method = getattr(obj, method_name, None)
    if method is None:
        return {"present": False}
    try:
        signature = str(inspect.signature(method))
    except (TypeError, ValueError):
        signature = None
    return {
        "present": True,
        "signature": signature,
        "source_location": source_location(method),
        "source_snippet": source_snippet(method, max_lines=max_lines),
    }


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _module_list(value: Any) -> list[Any]:
    if isinstance(value, nn.ModuleList):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def summarize_patch_transform(stage: Any, *, stage_index: int) -> dict[str, Any]:
    return {
        "stage_index": stage_index,
        "module_type": "PatchedPretransform",
        "class": class_summary(stage),
        "patch_size": _maybe_int(getattr(stage, "patch_size", None)),
        "downsample_ratio": _maybe_int(getattr(stage, "downsample_ratio", None)),
        "is_downsample": getattr(stage, "is_downsample", None),
        "declared_module_type": getattr(stage, "module_type", None),
        "methods": {
            name: method_summary(stage, name, max_lines=40)
            for name in ("encode", "decode", "forward")
        },
    }


def summarize_attention(attn: Any) -> dict[str, Any]:
    in_proj = _first_attr(attn, "in_proj", "qkv_proj", "c_attn")
    out_proj = _first_attr(attn, "out_proj", "proj", "c_proj")
    embed_dim = _maybe_int(_first_attr(attn, "embed_dim", "hidden_size"))
    num_heads = _maybe_int(_first_attr(attn, "num_heads", "n_heads"))
    head_dim = _maybe_int(_first_attr(attn, "head_dim", "head_size"))
    if head_dim is None and embed_dim is not None and num_heads:
        head_dim = embed_dim // num_heads
    return {
        "class": class_summary(attn),
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "causal": getattr(attn, "causal", None),
        "context": _maybe_int(getattr(attn, "context", None)),
        "max_period": _maybe_float(getattr(attn, "max_period", None)),
        "in_proj": linear_summary(in_proj) if in_proj is not None else None,
        "out_proj": linear_summary(out_proj) if out_proj is not None else None,
        "methods": {
            name: method_summary(attn, name, max_lines=80)
            for name in (
                "forward",
                "_forward_non_streaming_flash",
                "_forward_non_streaming_sdpa",
                "_project_qkv",
                "_apply_packed_rope",
                "_run_flash_attention",
            )
        },
    }


def summarize_transformer_layer(layer: Any, *, layer_index: int) -> dict[str, Any]:
    attn = _first_attr(layer, "self_attn", "attn")
    ffn = _first_attr(layer, "ffn", "mlp")
    ffn_modules = list(ffn) if isinstance(ffn, nn.Sequential) else []
    return {
        "layer_index": layer_index,
        "class": class_summary(layer),
        "norm1": norm_summary(_first_attr(layer, "norm1", "ln_1")),
        "norm2": norm_summary(_first_attr(layer, "norm2", "ln_2")),
        "layer_scale_1": tensor_summary(
            getattr(_first_attr(layer, "layer_scale_1"), "scale", None)
        ),
        "layer_scale_2": tensor_summary(
            getattr(_first_attr(layer, "layer_scale_2"), "scale", None)
        ),
        "self_attn": summarize_attention(attn) if attn is not None else None,
        "ffn": {
            "class": class_summary(ffn) if ffn is not None else None,
            "modules": [
                {
                    "index": index,
                    "class_name": module.__class__.__name__,
                    "linear": linear_summary(module),
                }
                for index, module in enumerate(ffn_modules)
            ],
        },
        "methods": {"forward": method_summary(layer, "forward", max_lines=60)},
    }


def summarize_transformer(transformer: Any) -> dict[str, Any]:
    layers = _module_list(getattr(transformer, "layers", None))
    first_layer = layers[0] if layers else None
    first_attn = _first_attr(first_layer, "self_attn", "attn") if first_layer else None
    return {
        "class": class_summary(transformer),
        "layer_count": len(layers),
        "positional_embedding": getattr(transformer, "positional_embedding", None),
        "max_period": _maybe_float(getattr(transformer, "max_period", None)),
        "positional_scale": _maybe_float(
            getattr(transformer, "positional_scale", None)
        ),
        "first_attention": (
            summarize_attention(first_attn) if first_attn is not None else None
        ),
        "first_layer": (
            summarize_transformer_layer(first_layer, layer_index=0)
            if first_layer is not None
            else None
        ),
        "methods": {
            name: method_summary(transformer, name, max_lines=80)
            for name in ("forward", "resolve_attention_implementation")
        },
    }


def summarize_projected_transformer(stage: Any, *, stage_index: int) -> dict[str, Any]:
    transformer = getattr(stage, "transformer", None)
    layers = _module_list(getattr(transformer, "layers", None)) if transformer else []
    first_layer = layers[0] if layers else None
    first_attn = _first_attr(first_layer, "self_attn", "attn") if first_layer else None
    first_attn_summary = (
        summarize_attention(first_attn) if first_attn is not None else None
    )
    input_proj = getattr(stage, "input_proj", None)
    output_proj = getattr(stage, "output_proj", None)
    first_ffn = _first_attr(first_layer, "ffn", "mlp") if first_layer else None
    ffn_modules = list(first_ffn) if isinstance(first_ffn, nn.Sequential) else []
    return {
        "stage_index": stage_index,
        "module_type": "Transformer",
        "class": class_summary(stage),
        "is_streaming": getattr(stage, "is_streaming", None),
        "input_proj": linear_summary(input_proj),
        "output_proj": linear_summary(output_proj),
        "input_dimension": _maybe_int(
            getattr(input_proj, "in_features", None) if input_proj is not None else None
        ),
        "output_dimension": _maybe_int(
            getattr(output_proj, "out_features", None)
            if output_proj is not None
            else None
        ),
        "d_model": _maybe_int(
            getattr(input_proj, "out_features", None)
            if input_proj is not None
            else None
        ),
        "layers": len(layers),
        "heads": first_attn_summary.get("num_heads") if first_attn_summary else None,
        "head_dim": first_attn_summary.get("head_dim") if first_attn_summary else None,
        "ffn": _maybe_int(
            getattr(ffn_modules[0], "out_features", None) if ffn_modules else None
        ),
        "causal": (
            getattr(first_attn, "causal", None) if first_attn is not None else None
        ),
        "context": (
            _maybe_int(getattr(first_attn, "context", None))
            if first_attn is not None
            else None
        ),
        "context_duration": _maybe_float(getattr(stage, "context_duration", None)),
        "transformer": (
            summarize_transformer(transformer) if transformer is not None else None
        ),
        "methods": {"forward": method_summary(stage, "forward", max_lines=80)},
    }


def summarize_decoder_stage(stage: Any, *, stage_index: int) -> dict[str, Any]:
    if hasattr(stage, "transformer"):
        return summarize_projected_transformer(stage, stage_index=stage_index)
    if hasattr(stage, "patch_size"):
        return summarize_patch_transform(stage, stage_index=stage_index)
    return {
        "stage_index": stage_index,
        "module_type": stage.__class__.__name__,
        "class": class_summary(stage),
        "methods": {"forward": method_summary(stage, "forward", max_lines=60)},
    }


def summarize_state_dict_groups(module: nn.Module) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "layers": set(), "examples": []}
    )
    for name, tensor in module.state_dict().items():
        match = _LAYER_PARAM_RE.match(name)
        if match:
            stage, layer, suffix = match.groups()
            key = f"{stage}.transformer.layers.{{layer}}.{suffix}"
            groups[key]["layers"].add(int(layer))
        else:
            parts = name.split(".")
            if len(parts) >= 2 and parts[0].isdigit():
                key = f"{parts[0]}.{parts[1]}.*"
            else:
                key = name
        group = groups[key]
        group["count"] += 1
        if len(group["examples"]) < 3:
            group["examples"].append(
                {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            )

    normalized: dict[str, Any] = {}
    for key in sorted(groups):
        group = groups[key]
        layers = sorted(group["layers"])
        normalized[key] = {
            "count": group["count"],
            "layers": layers,
            "layer_count": len(layers) if layers else None,
            "examples": group["examples"],
        }
    return normalized


def summarize_moss_tts_local_vocoder(processor: Any) -> dict[str, Any]:
    """Summarize the loaded MOSS-TTS Local processor and vocoder decoder."""
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is None:
        raise RuntimeError("processor.audio_tokenizer is required")
    decoder = getattr(audio_tokenizer, "decoder", None)
    decoder_stages = _module_list(decoder)
    stage_summaries = [
        summarize_decoder_stage(stage, stage_index=index)
        for index, stage in enumerate(decoder_stages)
    ]
    transformer_stages = [
        stage for stage in stage_summaries if stage.get("module_type") == "Transformer"
    ]
    return {
        "schema": "moss_tts_local_vocoder_introspection_v2",
        "processor": {
            "class": class_summary(processor),
            "model_config": _summarize_config(getattr(processor, "model_config", None)),
            "model_config_status": _summarize_config_source(
                "processor.model_config", getattr(processor, "model_config", None)
            ),
            "config_sources": _summarize_config_sources(processor, audio_tokenizer),
            "methods": {
                "decode_audio_codes": method_summary(
                    processor, "decode_audio_codes", max_lines=100
                )
            },
        },
        "audio_tokenizer": {
            "class": class_summary(audio_tokenizer),
            "config": _summarize_config(getattr(audio_tokenizer, "config", None)),
            "config_status": _summarize_config_source(
                "audio_tokenizer.config", getattr(audio_tokenizer, "config", None)
            ),
            "methods": {
                "_decode_frame": method_summary(
                    audio_tokenizer, "_decode_frame", max_lines=100
                ),
                "decode": method_summary(audio_tokenizer, "decode", max_lines=100),
            },
        },
        "decoder": {
            "class": class_summary(decoder) if decoder is not None else None,
            "stage_count": len(decoder_stages),
            "transformer_stage_count": len(transformer_stages),
            "transformer_layer_count": sum(
                int(stage.get("layers") or 0) for stage in transformer_stages
            ),
            "stages": stage_summaries,
            "state_dict_groups": (
                summarize_state_dict_groups(decoder)
                if isinstance(decoder, nn.Module)
                else {}
            ),
        },
    }


def _summarize_config(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    out: dict[str, Any] = {"class": class_summary(config)}
    for name in (
        "n_vq",
        "sampling_rate",
        "audio_vocab_size",
        "vocab_size",
        "num_codebooks",
        "codebook_size",
    ):
        if hasattr(config, name):
            value = getattr(config, name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[name] = value
    return out


def _summarize_config_source(label: str, config: Any) -> dict[str, Any]:
    summary = _summarize_config(config)
    return {
        "label": label,
        "present": config is not None,
        "class": class_summary(config) if config is not None else None,
        "values": summary or {},
    }


def _summarize_config_sources(
    processor: Any, audio_tokenizer: Any
) -> list[dict[str, Any]]:
    candidates = (
        ("processor.model_config", getattr(processor, "model_config", None)),
        ("processor.config", getattr(processor, "config", None)),
        ("audio_tokenizer.config", getattr(audio_tokenizer, "config", None)),
        (
            "audio_tokenizer.model_config",
            getattr(audio_tokenizer, "model_config", None),
        ),
    )
    return [_summarize_config_source(label, config) for label, config in candidates]


__all__ = [
    "class_summary",
    "linear_summary",
    "method_summary",
    "norm_summary",
    "source_location",
    "source_snippet",
    "summarize_decoder_stage",
    "summarize_moss_tts_local_vocoder",
    "summarize_state_dict_groups",
    "tensor_summary",
]
