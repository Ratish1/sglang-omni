"""Check existing batch-invariant projection arithmetic on real serving matrices."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from sglang.srt.batch_invariant_ops.batch_invariant_ops import matmul_persistent

from scripts.debug_qwen3_omni_projection_reduction_control import (
    ProjectionRuntime,
    ProjectionStep,
    restore_matrix,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    runtime = ProjectionRuntime.model_validate_json(
        (arguments.capture_directory / "projection-runtime.json").read_bytes()
    )
    assert runtime.preferred_blas_library == str(
        torch.backends.cuda.preferred_blas_library()
    )
    captured_weights = torch.load(
        arguments.capture_directory / "actual-projection-weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    weights = {
        name: restore_matrix(
            tensor, runtime.weight_strides[name], runtime.weight_storage_offsets[name]
        )
        for name, tensor in captured_weights.items()
    }
    counts: Counter[str] = Counter()
    arguments.output_directory.mkdir(exist_ok=False)
    with (
        (arguments.capture_directory / "steps.jsonl").open() as steps,
        (arguments.output_directory / "replay.jsonl").open("w") as output,
    ):
        for line in steps:
            metadata = ProjectionStep.model_validate_json(line)
            if metadata.phase != "MIXED":
                continue
            else:
                pass
            snapshots = torch.load(
                arguments.capture_directory / metadata.tensor_file,
                map_location="cpu",
                weights_only=True,
            )["full_projections"]
            for name, layout in metadata.full_projection_layouts.items():
                inputs = restore_matrix(
                    snapshots[f"{name}.input"],
                    [layout.row_stride_elements, layout.column_stride_elements],
                    layout.storage_offset_elements,
                )
                weight = weights[name]
                assert torch.equal(
                    F.linear(inputs, weight).cpu(), snapshots[f"{name}.output"]
                )
                grouped = matmul_persistent(inputs, weight.T)
                matmul_persistent(inputs[:1], weight.T)
                grouped_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(grouped_graph):
                    grouped_output = matmul_persistent(inputs, weight.T)
                single_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(single_graph):
                    single_output = torch.cat(
                        [
                            matmul_persistent(inputs[row : row + 1], weight.T)
                            for row in range(layout.rows)
                        ]
                    )
                grouped_graph.replay()
                single_graph.replay()
                assert torch.equal(grouped, grouped_output)
                before = single_output.clone()
                single_graph.replay()
                assert torch.equal(before, single_output)
                grouped_graph.replay()
                assert torch.equal(grouped, grouped_output)
                changed = grouped_output != single_output
                native_changes = grouped_output.cpu() != snapshots[f"{name}.output"]
                counts["matrix_calls"] += 1
                counts["physical_rows"] += layout.rows
                counts["changed_coordinates_between_geometries"] += int(changed.sum())
                counts["changed_coordinates_from_native"] += int(native_changes.sum())
                output.write(
                    json.dumps(
                        {
                            "tensor_file": metadata.tensor_file,
                            "boundary": name,
                            "physical_rows": layout.rows,
                            "changed_coordinates_between_geometries": int(
                                changed.sum()
                            ),
                            "changed_coordinates_from_native": int(
                                native_changes.sum()
                            ),
                            "eager_graph_and_repeat_exact": True,
                        }
                    )
                    + "\n"
                )
                output.flush()
    assert counts["matrix_calls"]
    report = {
        "counts": dict(counts),
        "reference": "sglang.srt.batch_invariant_ops.batch_invariant_ops.matmul_persistent",
        "scope": "Offline diagnostic on all actual MIXED rows. No global override, model generation, new precision default or speech quality claim.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
