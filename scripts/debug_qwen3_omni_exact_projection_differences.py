"""Resolve differing native projection coordinates with exact binary arithmetic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open

from scripts.debug_qwen3_omni_compare_thinker_decisions import (
    read_boundaries,
    read_decisions,
)
from scripts.debug_qwen3_omni_thinker_decisions import CheckpointIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-directory", type=Path, required=True)
    parser.add_argument("--second-directory", type=Path, required=True)
    parser.add_argument("--comparison-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    first_requests = read_decisions(arguments.first_directory)
    second_requests = read_decisions(arguments.second_directory)
    index = CheckpointIndex.model_validate_json(
        (arguments.checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    weights: dict[int, torch.Tensor] = {}
    counts: Counter[str] = Counter()
    maximum_midpoint_distance = 0.0
    arguments.output_directory.mkdir(exist_ok=False)
    with (
        (
            arguments.comparison_directory / "layer-differences.jsonl"
        ).open() as differences,
        (arguments.output_directory / "coordinates.jsonl").open("w") as output,
    ):
        for line in differences:
            difference = json.loads(line)
            names = difference["equal_query_input_changed_outputs"]
            if not names:
                continue
            else:
                pass
            request_id = difference["request_id"]
            position = difference["output_position"]
            first = first_requests[request_id][position]
            second = second_requests[request_id][position]
            assert (
                first.step.output_prefixes[first.row]
                == second.step.output_prefixes[second.row]
            )
            before = read_boundaries(arguments.first_directory, first.step)
            after = read_boundaries(arguments.second_directory, second.step)
            for name in names:
                assert name.endswith(".attention_projection.output"), name
                layer_index = int(name.split(".")[1])
                if layer_index not in weights:
                    weight_name = (
                        f"thinker.model.layers.{layer_index}.self_attn.o_proj.weight"
                    )
                    with safe_open(
                        arguments.checkpoint_directory / index.weight_map[weight_name],
                        framework="pt",
                        device="cpu",
                    ) as checkpoint:
                        weights[layer_index] = checkpoint.get_tensor(weight_name)
                else:
                    pass
                input_name = name.removesuffix(".output") + ".input"
                inputs = before[input_name][first.row]
                assert torch.equal(inputs, after[input_name][second.row])
                first_output = before[name][first.row]
                second_output = after[name][second.row]
                assert (
                    inputs.dtype
                    == weights[layer_index].dtype
                    == first_output.dtype
                    == second_output.dtype
                    == torch.bfloat16
                )
                counts["equal_input_changed_output_rows"] += 1
                coordinates = (
                    (first_output != second_output).nonzero().flatten().tolist()
                )
                input_ratios = [value.as_integer_ratio() for value in inputs.tolist()]
                for coordinate in coordinates:
                    weight_ratios = [
                        value.as_integer_ratio()
                        for value in weights[layer_index][coordinate].tolist()
                    ]
                    products = [
                        (a * c, b * d)
                        for (a, b), (c, d) in zip(
                            input_ratios, weight_ratios, strict=True
                        )
                    ]
                    denominator = max(divisor for _, divisor in products)
                    numerator = sum(
                        value * (denominator // divisor) for value, divisor in products
                    )
                    exact = Fraction(numerator, denominator)
                    first_value = float(first_output[coordinate])
                    second_value = float(second_output[coordinate])
                    first_fraction = Fraction(first_value)
                    second_fraction = Fraction(second_value)
                    approximate = torch.tensor(
                        float(exact), dtype=torch.float64
                    ).bfloat16()
                    candidates = [
                        torch.nextafter(
                            approximate, torch.tensor(-torch.inf, dtype=torch.bfloat16)
                        ),
                        approximate,
                        torch.nextafter(
                            approximate, torch.tensor(torch.inf, dtype=torch.bfloat16)
                        ),
                    ]
                    assert all(bool(candidate.isfinite()) for candidate in candidates)
                    nearest = min(
                        candidates,
                        key=lambda candidate: (
                            abs(exact - Fraction(float(candidate))),
                            int(candidate.view(torch.int16)) & 1,
                        ),
                    )
                    adjacent = (
                        abs(
                            int(first_output[coordinate].view(torch.int16))
                            - int(second_output[coordinate].view(torch.int16))
                        )
                        == 1
                    )
                    midpoint = (first_fraction + second_fraction) / 2
                    midpoint_distance = float(
                        abs(exact - midpoint) / abs(first_fraction - second_fraction)
                    )
                    counts["changed_coordinates"] += 1
                    counts["adjacent_bf16_outputs"] += adjacent
                    counts["exact_midpoints"] += exact == midpoint
                    counts["first_correctly_rounded"] += first_value == float(nearest)
                    counts["second_correctly_rounded"] += second_value == float(nearest)
                    counts["neither_correctly_rounded"] += first_value != float(
                        nearest
                    ) and second_value != float(nearest)
                    maximum_midpoint_distance = max(
                        maximum_midpoint_distance, midpoint_distance
                    )
                    output.write(
                        json.dumps(
                            {
                                "request_id": request_id,
                                "output_position": position,
                                "layer_index": layer_index,
                                "coordinate": coordinate,
                                "first": first_value,
                                "second": second_value,
                                "exact_numerator": str(exact.numerator),
                                "exact_denominator": str(exact.denominator),
                                "nearest_even_bf16": float(nearest),
                                "adjacent_outputs": adjacent,
                                "distance_from_pair_midpoint_in_pair_spacings": midpoint_distance,
                            }
                        )
                        + "\n"
                    )
    report = {
        "counts": dict(counts),
        "maximum_midpoint_distance_in_pair_spacings": maximum_midpoint_distance,
        "interpretation": "Exact sums establish coordinate rounding. BLAS does not promise correctly rounded, batch-invariant dot products.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
