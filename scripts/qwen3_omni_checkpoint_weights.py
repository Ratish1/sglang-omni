"""Verify loaded checkpoint references independently of conversion success flags."""

from __future__ import annotations

import inspect
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

import torch
from pydantic import BaseModel
from safetensors import safe_open
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeRotaryEmbedding,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    Qwen3OmniMoeVisionRotaryEmbedding,
    SinusoidsPositionEmbedding,
    eager_attention_forward,
)

from scripts.qwen3_omni_reference_observers import tensor_layout

EXPERT_WEIGHT = re.compile(
    r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)


class CheckpointIndex(BaseModel):
    weight_map: dict[str, str]


class LayerRuntime(TypedDict):
    layer_index: int
    expert_implementation: str | None
    expert_function: str
    attention_implementation: str | None
    attention_function: str


def record_reference_runtime(
    model: Qwen3OmniMoeForConditionalGeneration, output_file: Path
) -> None:
    layers: list[LayerRuntime] = []
    for layer_index, layer in enumerate(model.talker.model.layers):
        experts_module = layer.mlp.experts
        expert_function = ALL_EXPERTS_FUNCTIONS.get_interface(
            experts_module.config._experts_implementation,
            inspect.unwrap(experts_module.forward),
        )
        attention_function = ALL_ATTENTION_FUNCTIONS.get_interface(
            layer.self_attn.config._attn_implementation, eager_attention_forward
        )
        layers.append(
            {
                "layer_index": layer_index,
                "expert_implementation": experts_module.config._experts_implementation,
                "expert_function": f"{expert_function.__module__}.{expert_function.__qualname__}",
                "attention_implementation": layer.self_attn.config._attn_implementation,
                "attention_function": f"{attention_function.__module__}.{attention_function.__qualname__}",
            }
        )
    report = {
        "talker_layers": layers,
        "talker_parameter_layouts": {
            name: tensor_layout(parameter)
            for name, parameter in model.talker.named_parameters()
        },
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_bfloat16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "allow_tf32_convolutions": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment_controls": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "NVIDIA_TF32_OVERRIDE",
            )
        },
    }
    output_file.write_text(json.dumps(report, indent=2))


def audit_checkpoint_buffers(
    model: Qwen3OmniMoeForConditionalGeneration,
    output_prefix: Path,
) -> None:
    expected_buffers: dict[str, torch.Tensor] = {}
    for module_path, module in model.named_modules():
        if isinstance(
            module,
            (
                Qwen3OmniMoeRotaryEmbedding,
                Qwen3OmniMoeThinkerTextRotaryEmbedding,
                Qwen3OmniMoeVisionRotaryEmbedding,
            ),
        ):
            reference_rotary = type(module)(module.config)
            expected_buffers[f"{module_path}.inv_freq"] = reference_rotary.inv_freq
            expected_buffers[f"{module_path}.original_inv_freq"] = (
                reference_rotary.original_inv_freq
            )
        elif isinstance(module, SinusoidsPositionEmbedding):
            position_embeddings = module.compute_default_singular_positional_embedding()
            if module.positional_embedding.dtype == model.dtype:
                position_embeddings = position_embeddings.to(model.dtype)
            else:
                pass
            expected_buffers[f"{module_path}.positional_embedding"] = (
                position_embeddings
            )
        elif isinstance(module, Qwen3OmniMoeCode2Wav):
            expected_buffers[f"{module_path}.code_offset"] = (
                torch.arange(module.config.num_quantizers).view(1, -1, 1)
                * module.config.codebook_size
            )
        else:
            pass
    actual_buffers = dict(model.named_buffers())
    assert actual_buffers.keys() == expected_buffers.keys()
    reports = []
    buffer_operands: dict[str, torch.Tensor] = {}
    for buffer_path, expected in expected_buffers.items():
        actual = actual_buffers[buffer_path].detach().cpu()
        expected = expected.detach().cpu()
        buffer_operands[f"{buffer_path}.actual"] = actual
        buffer_operands[f"{buffer_path}.expected"] = expected
        reports.append(
            {
                "buffer": buffer_path,
                "dtype": str(actual.dtype),
                "expected_dtype": str(expected.dtype),
                "exact_config": actual.dtype == expected.dtype
                and torch.equal(actual, expected),
                "different_values": int((actual != expected).sum()),
                "maximum_absolute_error": float(
                    (actual.double() - expected.double()).abs().max()
                ),
            }
        )
    output_prefix.with_suffix(".json").write_text(json.dumps(reports, indent=2))
    torch.save(buffer_operands, output_prefix.with_suffix(".pt"))


@torch.inference_mode()
def audit_checkpoint_weights(
    model: (
        Qwen3OmniMoeForConditionalGeneration
        | Qwen3OmniMoeTalkerForConditionalGeneration
        | Qwen3OmniMoeThinkerForConditionalGeneration
        | Qwen3OmniMoeCode2Wav
    ),
    checkpoint_directory: Path,
    output_file: Path,
) -> None:
    index = CheckpointIndex.model_validate_json(
        (checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    if isinstance(model, Qwen3OmniMoeTalkerForConditionalGeneration):
        prefix = "talker."
    elif isinstance(model, Qwen3OmniMoeCode2Wav):
        prefix = "code2wav."
    elif isinstance(model, Qwen3OmniMoeThinkerForConditionalGeneration):
        prefix = "thinker."
    else:
        prefix = ""
    tensors = model.state_dict(keep_vars=True)
    shards: dict[str, list[str]] = {}
    for checkpoint_name, shard in index.weight_map.items():
        if checkpoint_name.startswith(prefix):
            shards.setdefault(shard, []).append(checkpoint_name)
        else:
            pass
    visited: set[int] = set()
    mismatches: list[str] = []
    dtypes: Counter[str] = Counter()
    checked = 0
    for shard, names in sorted(shards.items()):
        with safe_open(
            checkpoint_directory / shard, framework="pt", device="cpu"
        ) as checkpoint:
            for checkpoint_name in names:
                expected = checkpoint.get_tensor(checkpoint_name)
                name = checkpoint_name.removeprefix(prefix)
                expert = EXPERT_WEIGHT.fullmatch(name)
                if expert is not None:
                    expert_prefix, expert_index, projection = expert.groups()
                    if projection == "down_proj":
                        tensor = tensors[f"{expert_prefix}.down_proj"]
                        actual = tensor[int(expert_index)]
                    else:
                        tensor = tensors[f"{expert_prefix}.gate_up_proj"]
                        rows = expected.shape[0]
                        start = 0 if projection == "gate_proj" else rows
                        actual = tensor[int(expert_index), start : start + rows]
                else:
                    tensor = tensors[name]
                    actual = tensor
                visited.add(id(tensor))
                dtypes[f"{expected.dtype}->{actual.dtype}"] += 1
                if actual.shape != expected.shape or not torch.equal(
                    actual, expected.to(actual.device)
                ):
                    mismatches.append(checkpoint_name)
                else:
                    pass
                checked += 1
    unvisited = sorted(
        name for name, tensor in tensors.items() if id(tensor) not in visited
    )
    device_properties = torch.cuda.get_device_properties(model.device)
    report = {
        "checkpoint_prefix": prefix,
        "checked_checkpoint_tensors": checked,
        "mismatches": mismatches,
        "unvisited_tensors": unvisited,
        "dtype_pairs": dict(dtypes),
        "torch_version": str(torch.__version__),
        "cuda_device": device_properties.name,
        "multiprocessors": device_properties.multi_processor_count,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bfloat16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    output_file.write_text(json.dumps(report, indent=2))
    assert not mismatches and not unvisited, report
