"""Verify native unquantized Thinker parameter views against checkpoint tensors."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

import torch
from pydantic import BaseModel
from safetensors import safe_open

logger = logging.getLogger(__name__)

QKV_PROJECTION = re.compile(r"(.+\.self_attn)\.([qkv])_proj\.(weight|bias)")
EXPERT_PROJECTION = re.compile(
    r"(.+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)
DENSE_PROJECTION = re.compile(r"(.+\.mlp)\.(gate_proj|up_proj)\.weight")


class CheckpointIndex(BaseModel):
    weight_map: dict[str, str]


def audit_thinker_weights(
    parameters: dict[str, torch.nn.Parameter],
    checkpoint_directory: Path,
    output_file: Path,
) -> None:
    index = CheckpointIndex.model_validate_json(
        (checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    shards: dict[str, list[str]] = {}
    for name, shard in index.weight_map.items():
        if name.startswith(("thinker.model.", "thinker.lm_head.")):
            shards.setdefault(shard, []).append(name)
        else:
            pass
    checked = 0
    visited: set[str] = set()
    mismatches: list[str] = []
    dtypes: Counter[str] = Counter()
    for shard, names in sorted(shards.items()):
        with safe_open(
            checkpoint_directory / shard, framework="pt", device="cpu"
        ) as checkpoint:
            for checkpoint_name in names:
                expected = checkpoint.get_tensor(checkpoint_name)
                name = checkpoint_name.removeprefix("thinker.")
                qkv = QKV_PROJECTION.fullmatch(name)
                expert = EXPERT_PROJECTION.fullmatch(name)
                dense = DENSE_PROJECTION.fullmatch(name)
                if qkv is not None:
                    prefix, projection, suffix = qkv.groups()
                    native_name = f"{prefix}.qkv_proj.{suffix}"
                    parameter = parameters[native_name]
                    rows = expected.shape[0]
                    start = (
                        0
                        if projection == "q"
                        else parameter.shape[0] - (2 if projection == "k" else 1) * rows
                    )
                    actual = parameter[start : start + rows]
                elif expert is not None:
                    prefix, expert_index, projection = expert.groups()
                    if projection == "down_proj":
                        native_name = f"{prefix}.w2_weight"
                        actual = parameters[native_name][int(expert_index)]
                    else:
                        native_name = f"{prefix}.w13_weight"
                        rows = expected.shape[0]
                        start = 0 if projection == "gate_proj" else rows
                        actual = parameters[native_name][
                            int(expert_index), start : start + rows
                        ]
                elif dense is not None:
                    prefix, projection = dense.groups()
                    native_name = f"{prefix}.gate_up_proj.weight"
                    rows = expected.shape[0]
                    start = 0 if projection == "gate_proj" else rows
                    actual = parameters[native_name][start : start + rows]
                else:
                    native_name = name
                    actual = parameters[native_name]
                visited.add(native_name)
                dtypes[f"{expected.dtype}->{actual.dtype}"] += 1
                if actual.shape != expected.shape or not torch.equal(
                    actual, expected.to(actual.device)
                ):
                    mismatches.append(checkpoint_name)
                else:
                    pass
                checked += 1
    unvisited = sorted(parameters.keys() - visited)
    report = {
        "checked_checkpoint_tensors": checked,
        "native_parameters": len(parameters),
        "visited_native_parameters": len(visited),
        "mismatches": mismatches,
        "unvisited_native_parameters": unvisited,
        "dtype_pairs": dict(dtypes),
    }
    output_file.write_text(json.dumps(report, indent=2))
    assert not mismatches and not unvisited, report
    logger.info(
        f"Verified {checked} Thinker checkpoint tensors in {len(visited)} native parameters"
    )
