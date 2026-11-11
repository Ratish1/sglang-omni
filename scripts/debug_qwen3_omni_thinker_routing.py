"""Check actual Thinker routing across complete captured token lifetimes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from scripts.debug_qwen3_omni_compare_thinker_decisions import CapturedStep
from scripts.debug_qwen3_omni_thinker_decisions import CheckpointIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = AutoConfig.from_pretrained(
        arguments.checkpoint_directory
    ).thinker_config.text_config
    assert configuration.norm_topk_prob
    index = CheckpointIndex.model_validate_json(
        (arguments.checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    weights: dict[int, torch.Tensor] = {}
    for name, filename in index.weight_map.items():
        if name.startswith("thinker.model.layers.") and name.endswith(
            ".mlp.gate.weight"
        ):
            layer_index = int(name.split(".")[3])
            with safe_open(
                arguments.checkpoint_directory / filename, framework="pt", device="cpu"
            ) as checkpoint:
                weights[layer_index] = checkpoint.get_tensor(name).to(
                    "cuda", torch.float64
                )
        else:
            pass
    assert set(weights) == set(range(configuration.num_hidden_layers))
    arguments.output_directory.mkdir(exist_ok=False)
    counts: Counter[str] = Counter()
    maximum_projection_error = 0.0
    maximum_weight_error = 0.0
    with (
        (arguments.capture_directory / "steps.jsonl").open() as steps,
        (arguments.output_directory / "routing.jsonl").open("w") as output,
        torch.inference_mode(),
    ):
        for line in steps:
            metadata = CapturedStep.model_validate_json(line)
            boundaries = torch.load(
                arguments.capture_directory / metadata.tensor_file,
                map_location="cpu",
                weights_only=True,
            )["layer_boundaries"]
            live_rows = len(metadata.request_ids)
            assert len(boundaries) == len(weights) * 10
            for layer_index, weight in weights.items():
                prefix = f"layer.{layer_index}"
                inputs = boundaries[f"{prefix}.gate.input"].to("cuda", torch.float64)
                logits = boundaries[f"{prefix}.gate.output"].to("cuda", torch.float64)
                expert_ids = boundaries[f"{prefix}.expert_ids"].to("cuda")
                expert_weights = boundaries[f"{prefix}.expert_weights"].to(
                    "cuda", torch.float64
                )
                assert (
                    inputs.shape[0]
                    == logits.shape[0]
                    == expert_ids.shape[0]
                    == live_rows
                )
                assert bool(
                    inputs.isfinite().all()
                    & logits.isfinite().all()
                    & expert_weights.isfinite().all()
                )
                assert bool((expert_weights >= 0).all())
                sorted_ids = expert_ids.sort(-1).values
                assert bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all())
                selected_logits = logits.gather(1, expert_ids)
                cutoff = selected_logits.min(-1).values
                others = logits.clone().scatter_(1, expert_ids, -torch.inf)
                assert bool((cutoff >= others.max(-1).values).all())
                tied_cutoff = cutoff == others.max(-1).values
                expected_logits = inputs @ weight.T
                assert bool(expected_logits.isfinite().all())
                reference_ids = (
                    expected_logits.topk(expert_ids.shape[-1], dim=-1)
                    .indices.sort(-1)
                    .values
                )
                changed_set = (reference_ids != sorted_ids).any(-1)
                rounded_equal = (
                    expected_logits.to(torch.bfloat16).double() == logits
                ).all(-1)
                projection_errors = (expected_logits - logits).abs().max(-1).values
                expected_weights = selected_logits.softmax(-1)
                weight_errors = (expected_weights - expert_weights).abs().max(-1).values
                maximum_projection_error = max(
                    maximum_projection_error, float(projection_errors.max())
                )
                maximum_weight_error = max(
                    maximum_weight_error, float(weight_errors.max())
                )
                tied_cutoff = tied_cutoff.cpu()
                changed_set = changed_set.cpu()
                rounded_equal = rounded_equal.cpu()
                projection_errors = projection_errors.cpu()
                weight_errors = weight_errors.cpu()
                for row, request_id in enumerate(metadata.request_ids):
                    counts["row_layer_decisions"] += 1
                    counts[f"phase_{metadata.phase}"] += 1
                    counts["native_cutoff_ties"] += bool(tied_cutoff[row])
                    counts["fp64_changes_set"] += bool(changed_set[row])
                    counts["changes_without_native_ties"] += bool(
                        changed_set[row] & ~tied_cutoff[row]
                    )
                    counts["projection_equal_rounded_fp64"] += bool(rounded_equal[row])
                    output.write(
                        json.dumps(
                            {
                                "request_id": request_id,
                                "output_position": len(metadata.output_prefixes[row]),
                                "layer_index": layer_index,
                                "phase": metadata.phase,
                                "native_cutoff_tied": bool(tied_cutoff[row]),
                                "fp64_changes_set": bool(changed_set[row]),
                                "projection_equal_rounded_fp64": bool(
                                    rounded_equal[row]
                                ),
                                "maximum_logit_error": float(projection_errors[row]),
                                "maximum_weight_error": float(weight_errors[row]),
                            }
                        )
                        + "\n"
                    )
    report = {
        "counts": dict(counts),
        "layers": len(weights),
        "maximum_projection_error": maximum_projection_error,
        "maximum_weight_error": maximum_weight_error,
        "interpretation": "Actual top-k validity is asserted separately from FP64 rounding sensitivity.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
