"""Measure actual Thinker attention projections against checkpoint arithmetic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from scripts.debug_qwen3_omni_compare_thinker_decisions import (
    CapturedStep,
    read_boundaries,
)
from scripts.debug_qwen3_omni_thinker_decisions import CheckpointIndex


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = AutoConfig.from_pretrained(
        arguments.checkpoint_directory
    ).thinker_config.text_config
    assert not configuration.attention_bias
    index = CheckpointIndex.model_validate_json(
        (arguments.checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    weights: dict[str, torch.Tensor] = {}
    for layer_index in range(configuration.num_hidden_layers):
        prefix = f"thinker.model.layers.{layer_index}.self_attn."
        components: dict[str, torch.Tensor] = {}
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = prefix + projection + ".weight"
            with safe_open(
                arguments.checkpoint_directory / index.weight_map[name],
                framework="pt",
                device="cpu",
            ) as checkpoint:
                components[projection] = checkpoint.get_tensor(name).to(
                    "cuda", torch.float64
                )
        weights[f"layer.{layer_index}.qkv"] = torch.cat(
            [components[name] for name in ("q_proj", "k_proj", "v_proj")]
        )
        weights[f"layer.{layer_index}.attention_projection"] = components["o_proj"]
    arguments.output_directory.mkdir(exist_ok=False)
    counts: dict[str, Counter[str]] = {name: Counter() for name in weights}
    maximum_errors: dict[str, float] = {name: 0.0 for name in weights}
    maximum_relative_errors: dict[str, float] = {name: 0.0 for name in weights}
    with (
        (arguments.capture_directory / "steps.jsonl").open() as steps,
        (arguments.output_directory / "projections.jsonl").open("w") as output,
    ):
        for line in steps:
            metadata = CapturedStep.model_validate_json(line)
            boundaries = read_boundaries(arguments.capture_directory, metadata)
            for name, weight in weights.items():
                inputs = boundaries[f"{name}.input"].to("cuda", torch.float64)
                actual = boundaries[f"{name}.output"].to("cuda", torch.float64)
                assert inputs.shape[0] == actual.shape[0] == len(metadata.request_ids)
                assert inputs.shape[-1] == weight.shape[-1]
                assert actual.shape[-1] == weight.shape[0]
                expected = inputs @ weight.T
                assert bool(
                    inputs.isfinite().all()
                    & actual.isfinite().all()
                    & expected.isfinite().all()
                )
                differences = actual - expected
                relative_errors = differences.norm(dim=-1) / expected.norm(
                    dim=-1
                ).clamp_min(torch.finfo(torch.float64).tiny)
                absolute_errors = differences.abs().max(-1).values
                rounded_equal = (actual == expected.to(torch.bfloat16).double()).all(-1)
                maximum_errors[name] = max(
                    maximum_errors[name], float(absolute_errors.max())
                )
                maximum_relative_errors[name] = max(
                    maximum_relative_errors[name], float(relative_errors.max())
                )
                relative_errors = relative_errors.cpu()
                absolute_errors = absolute_errors.cpu()
                rounded_equal = rounded_equal.cpu()
                for row, request_id in enumerate(metadata.request_ids):
                    counts[name]["rows"] += 1
                    counts[name][f"phase_{metadata.phase}"] += 1
                    counts[name]["equal_rounded_fp64"] += bool(rounded_equal[row])
                    output.write(
                        json.dumps(
                            {
                                "request_id": request_id,
                                "output_position": len(metadata.output_prefixes[row]),
                                "boundary": name,
                                "phase": metadata.phase,
                                "equal_rounded_fp64": bool(rounded_equal[row]),
                                "maximum_absolute_error": float(absolute_errors[row]),
                                "relative_l2_error": float(relative_errors[row]),
                            }
                        )
                        + "\n"
                    )
    report = {
        "boundaries": {
            name: {
                **counter,
                "maximum_absolute_error": maximum_errors[name],
                "maximum_relative_l2_error": maximum_relative_errors[name],
            }
            for name, counter in counts.items()
        },
        "interpretation": "Independent projection error is measured without imposing a serving tolerance or changing dtype/backend.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {
                "projection_boundaries": len(weights),
                "rows": sum(counter["rows"] for counter in counts.values()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
