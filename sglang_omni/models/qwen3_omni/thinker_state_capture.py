"""Record native checkpoint layout and expert execution without changing either."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict

import torch
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel

EXPERT_PARAMETER = re.compile(r"layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)")
QUERY_KEY_VALUE_PARAMETER = re.compile(r"layers\.(\d+)\.self_attn\.qkv_proj\.weight")


class ParameterIdentity(TypedDict):
    shape: list[int]
    dtype: str
    sha256: str


@torch.inference_mode()
def record_thinker_state(
    model: Qwen3MoeLLMModel, head: torch.nn.Module, directory: Path
) -> None:
    parameters: dict[str, ParameterIdentity] = {}
    for name, parameter in list(model.named_parameters()) + [
        ("lm_head.weight", head.weight)
    ]:
        actual = parameter.detach().cpu()
        expert = EXPERT_PARAMETER.fullmatch(name)
        projection = QUERY_KEY_VALUE_PARAMETER.fullmatch(name)
        unpacked: dict[str, torch.Tensor] = {}
        if expert is not None:
            layer_index, kind = expert.groups()
            experts = model.layers[int(layer_index)].mlp.experts
            assert isinstance(experts.quant_method, UnquantizedFusedMoEMethod)
            assert experts.runner_backend.is_triton()
            assert not experts.quant_method.use_triton_kernels
            for expert_index, weight in enumerate(actual):
                prefix = (
                    f"thinker.model.layers.{layer_index}.mlp.experts.{expert_index}"
                )
                if kind == "w2_weight":
                    unpacked[f"{prefix}.down_proj.weight"] = weight
                elif experts.quant_method.w13_swiglu_interleaved:
                    unpacked[f"{prefix}.gate_proj.weight"] = weight[0::2]
                    unpacked[f"{prefix}.up_proj.weight"] = weight[1::2]
                else:
                    gate, up = weight.chunk(2, dim=0)
                    unpacked[f"{prefix}.gate_proj.weight"] = gate
                    unpacked[f"{prefix}.up_proj.weight"] = up
        elif projection is not None:
            layer_index = projection.group(1)
            attention = model.layers[int(layer_index)].self_attn
            query, key, value = actual.split(
                (attention.q_size, attention.kv_size, attention.kv_size), dim=0
            )
            prefix = f"thinker.model.layers.{layer_index}.self_attn"
            unpacked[f"{prefix}.q_proj.weight"] = query
            unpacked[f"{prefix}.k_proj.weight"] = key
            unpacked[f"{prefix}.v_proj.weight"] = value
        elif name == "lm_head.weight":
            unpacked[f"thinker.{name}"] = actual
        else:
            unpacked[f"thinker.model.{name}"] = actual
        for checkpoint_name, weight in unpacked.items():
            assert checkpoint_name not in parameters
            parameters[checkpoint_name] = {
                "shape": list(weight.shape),
                "dtype": str(weight.dtype),
                "sha256": hashlib.sha256(
                    weight.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
    layers = []
    for layer_index, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        configuration = experts.moe_runner_config
        rotary = layer.self_attn.rotary_emb
        layers.append(
            {
                "layer_index": layer_index,
                "expert_backend": str(experts.runner_backend),
                "expert_method": type(experts.quant_method).__name__,
                "activation": configuration.activation,
                "is_gated": configuration.is_gated,
                "apply_router_weight_on_input": configuration.apply_router_weight_on_input,
                "no_combine": configuration.no_combine,
                "silu_mul_keep_fp32": configuration.silu_mul_keep_fp32,
                "w13_swiglu_interleaved": experts.quant_method.w13_swiglu_interleaved,
                "rotary_type": type(rotary).__name__,
                "rotary_implementation": rotary._forward_method.__qualname__,  # noqa: leading-underscore  # Runtime dispatch attribute.
                "mrope_section": rotary.mrope_section,
                "mrope_interleaved": rotary.mrope_interleaved,
                "mrope_interleaved_glm": rotary.mrope_interleaved_glm,
                "is_neox_style": rotary.is_neox_style,
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "native-thinker-state.json").write_text(
        json.dumps({"parameters": parameters, "layers": layers}, indent=2)
    )
