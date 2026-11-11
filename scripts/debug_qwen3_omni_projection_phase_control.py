"""Link cross-phase projection witnesses to their exact single-row reference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_persistent

from scripts.debug_qwen3_omni_compare_thinker_decisions import (
    read_boundaries,
    read_decisions,
)
from scripts.debug_qwen3_omni_thinker_decisions import CheckpointIndex


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-directory", type=Path, required=True)
    parser.add_argument("--second-directory", type=Path, required=True)
    parser.add_argument("--comparison-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--batch-invariant-reference", action="store_true")
    arguments = parser.parse_args()
    first_requests = read_decisions(arguments.first_directory)
    second_requests = read_decisions(arguments.second_directory)
    index = CheckpointIndex.model_validate_json(
        (arguments.checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    weights: dict[int, torch.Tensor] = {}
    captured_rows: dict[str, list[torch.Tensor]] = {
        "inputs": [],
        "first": [],
        "second": [],
        "single_row": [],
    }
    counts: Counter[str] = Counter()
    arguments.output_directory.mkdir(exist_ok=False)
    with (
        (
            arguments.comparison_directory / "layer-differences.jsonl"
        ).open() as differences,
        (arguments.output_directory / "witnesses.jsonl").open("w") as output,
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
                assert name.endswith(".attention_projection.output")
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
                        weights[layer_index] = checkpoint.get_tensor(weight_name).cuda()
                else:
                    pass
                inputs = before[name.removesuffix(".output") + ".input"][first.row]
                assert torch.equal(
                    inputs, after[name.removesuffix(".output") + ".input"][second.row]
                )
                reference_inputs = inputs.unsqueeze(0).cuda()
                if arguments.batch_invariant_reference:
                    reference = matmul_persistent(
                        reference_inputs, weights[layer_index].T
                    )[0].cpu()
                else:
                    reference = F.linear(reference_inputs, weights[layer_index])[
                        0
                    ].cpu()
                first_output = before[name][first.row]
                second_output = after[name][second.row]
                matches_first = torch.equal(reference, first_output)
                matches_second = torch.equal(reference, second_output)
                assert not torch.equal(first_output, second_output)
                counts["witnesses"] += 1
                counts[f"phase_pair_{first.step.phase}_{second.step.phase}"] += 1
                counts["matches_first"] += matches_first
                counts["matches_second"] += matches_second
                counts["matches_neither"] += not matches_first and not matches_second
                counts[f"matches_{first.step.phase}"] += matches_first
                counts[f"matches_{second.step.phase}"] += matches_second
                for field, tensor in (
                    ("inputs", inputs),
                    ("first", first_output),
                    ("second", second_output),
                    ("single_row", reference),
                ):
                    captured_rows[field].append(tensor.clone())
                output.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "output_position": position,
                            "layer_index": layer_index,
                            "first_phase": first.step.phase,
                            "second_phase": second.step.phase,
                            "first_source_tensor": first.step.tensor_file,
                            "second_source_tensor": second.step.tensor_file,
                            "matches_first": matches_first,
                            "matches_second": matches_second,
                            "first_changed_coordinates": int(
                                (reference != first_output).sum()
                            ),
                            "second_changed_coordinates": int(
                                (reference != second_output).sum()
                            ),
                        }
                    )
                    + "\n"
                )
    assert counts["witnesses"]
    torch.save(
        {
            "rows": {
                name: torch.stack(tensors) for name, tensors in captured_rows.items()
            },
            "weights": {str(layer): weight.cpu() for layer, weight in weights.items()},
        },
        arguments.output_directory / "reusable-witness-rows.pt",
    )
    report = {
        "counts": dict(counts),
        "reference": (
            "sglang.matmul_persistent"
            if arguments.batch_invariant_reference
            else "torch.nn.functional.linear"
        ),
        "scope": "Exact original query rows and raw checkpoint weights. Reusable tensors are query-level math witnesses, not complete padded GEMM operands.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
