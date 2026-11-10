"""Verify loaded checkpoint references independently of conversion success flags."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch
from pydantic import BaseModel
from safetensors import safe_open
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

EXPERT_WEIGHT = re.compile(
    r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)


class CheckpointIndex(BaseModel):
    weight_map: dict[str, str]


@torch.inference_mode()
def audit_checkpoint_weights(
    model: (
        Qwen3OmniMoeForConditionalGeneration
        | Qwen3OmniMoeTalkerForConditionalGeneration
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
