# SPDX-License-Identifier: Apache-2.0
"""Check all Qwen3-Omni checkpoint scales against a double-precision oracle."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from sglang_omni.quantization import get_weight_preprocessor


@dataclass
class ScaleCounts:
    tensors: int = 0
    scales: int = 0
    old_rounded_scales: int = 0
    current_rounded_scales: int = 0
    old_max_relative_error: float = 0.0
    current_max_relative_error: float = 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    torch.set_num_threads(1)

    checkpoint = arguments.checkpoint
    matrix_shards = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    scale_fields_by_shard: dict[str, list[str]] = defaultdict(list)
    for field, shard in matrix_shards.items():
        if field.endswith(".weight_scale_inv"):
            scale_fields_by_shard[shard].append(field)
        else:
            pass
    assert scale_fields_by_shard
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    preprocess = get_weight_preprocessor(config, fp8_scale_inverted=True)
    all_counts: dict[str, ScaleCounts] = defaultdict(ScaleCounts)
    for shard, fields in sorted(scale_fields_by_shard.items()):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as tensors:
            for field in sorted(fields):
                stored_inverses = tensors.get_tensor(field)
                status = all_counts[f"{field.split('.')[0]}:{stored_inverses.dtype}"]
                ideal_scales = (1.0 / stored_inverses.double()).float()
                old_scales = stored_inverses.reciprocal().float()
                current_scales = preprocess(field, stored_inverses).float()
                status.tensors += 1
                status.scales += stored_inverses.numel()
                status.old_rounded_scales += int((old_scales != ideal_scales).sum())
                status.current_rounded_scales += int(
                    (current_scales != ideal_scales).sum()
                )
                status.old_max_relative_error = max(
                    status.old_max_relative_error,
                    float(((old_scales - ideal_scales) / ideal_scales).abs().max()),
                )
                status.current_max_relative_error = max(
                    status.current_max_relative_error,
                    float(((current_scales - ideal_scales) / ideal_scales).abs().max()),
                )

    arguments.output.write_text(
        json.dumps(
            {stage: asdict(counts) for stage, counts in all_counts.items()}, indent=2
        )
    )
    assert all(counts.old_rounded_scales for counts in all_counts.values())
    assert all(not counts.current_rounded_scales for counts in all_counts.values())


if __name__ == "__main__":
    main()
