# SPDX-License-Identifier: Apache-2.0
"""Validate one query restoration against its two native reference trajectories."""

import argparse
import json
from dataclasses import asdict
from hashlib import file_digest
from pathlib import Path

import torch
from score_qwen3_omni_fixed_conditioning import compare_conditioning


def main(
    baseline: Path, eager: Path, intervention: Path, original_results: Path
) -> None:
    torch.set_num_threads(1)
    selection = json.loads(
        (intervention / "captures/query-restore-selection.json").read_text()
    )
    sample_id = selection["request_id"].removeprefix("feedback-trace-native-")
    frame_index = selection["frame_index"]
    layer_index = selection["layer_index"]
    fused_comparisons = compare_conditioning(baseline, intervention, original_results)
    eager_comparisons = compare_conditioning(eager, intervention, original_results)
    for comparison in fused_comparisons:
        if comparison.sample_id == sample_id:
            assert comparison.first_differences
            assert min(comparison.first_differences.values()) == frame_index
            assert (
                comparison.first_differences[f"layer{layer_index}_query_before_rope"]
                == frame_index
            )
        else:
            assert not comparison.first_differences
            assert comparison.frame_counts[0] == comparison.frame_counts[1]
    assert sum(report.sample_id == sample_id for report in fused_comparisons) == 1
    frames: list[dict[str, torch.Tensor]] = []
    capture_sha256: dict[str, str] = {}
    for root in (baseline, eager, intervention):
        capture = root / "captures" / sample_id
        path = capture / "trajectory.pt"
        frames.append(
            torch.load(path, weights_only=True, map_location="cpu")[frame_index]
        )
        state = json.loads((capture / "trajectory.json").read_text())["frames"][
            frame_index
        ]
        assert state["cuda_graph"]
        if root == intervention:
            assert selection["request_row"] == state["request_row"]
        else:
            pass
        with path.open("rb") as stream:
            capture_sha256[str(path)] = file_digest(stream, "sha256").hexdigest()
    before, reference, actual = frames
    for earlier_layer in range(layer_index):
        prefix = f"layer{earlier_layer}_"
        for name in before:
            if name.startswith(prefix) and "_peer_" not in name:
                assert torch.equal(actual[name], before[name]), name
            else:
                pass
    prefix = f"layer{layer_index}_"
    for name in (
        "contribution",
        "previous_residual",
        "attention_normalized",
        "attention_residual",
        "qkv_projection",
        "key_before_rope",
        "value_before_cache",
    ):
        assert torch.equal(actual[prefix + name], before[prefix + name]), name
    assert torch.equal(
        actual[prefix + "qkv_projection"], reference[prefix + "qkv_projection"]
    )
    assert torch.equal(
        actual[prefix + "query_before_rope"], reference[prefix + "query_before_rope"]
    )
    changed_coordinates = (
        (before[prefix + "query_before_rope"] != actual[prefix + "query_before_rope"])
        .nonzero()
        .tolist()
    )
    assert changed_coordinates
    report = {
        "selection": selection,
        "capture_sha256": capture_sha256,
        "changed_query_coordinates": changed_coordinates,
        "checks": {
            "native_validation_and_held_conditioning": True,
            "fused_prefix_exact": True,
            "untouched_control_trajectories_exact": True,
            "earlier_layers_exact": True,
            "projected_query_key_value_exact": True,
            "restored_query_equals_eager": True,
            "key_and_value_unchanged": True,
            "native_cuda_graph_used": True,
        },
        "against_fused": [asdict(comparison) for comparison in fused_comparisons],
        "against_eager": [asdict(comparison) for comparison in eager_comparisons],
        "scope": "One selected native query intervention; acoustic scoring and population qualification are separate.",
    }
    output = intervention / "query-restore-audit.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                "checks": report["checks"],
                "changed_query_coordinates": changed_coordinates,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("eager", type=Path)
    parser.add_argument("intervention", type=Path)
    parser.add_argument("original_results", type=Path)
    arguments = parser.parse_args()
    main(
        arguments.baseline,
        arguments.eager,
        arguments.intervention,
        arguments.original_results,
    )
else:
    pass
