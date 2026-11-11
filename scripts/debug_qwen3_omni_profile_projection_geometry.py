"""Profile grouped and single-row projection kernels on captured serving operands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.debug_qwen3_omni_projection_reduction_control import (
    ProjectionRuntime,
    ProjectionStep,
    restore_matrix,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--tensor-file", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    runtime = ProjectionRuntime.model_validate_json(
        (arguments.capture_directory / "projection-runtime.json").read_bytes()
    )
    assert runtime.bf16_gemm_backend in ("auto", "torch")
    assert runtime.preferred_blas_library == str(
        torch.backends.cuda.preferred_blas_library()
    )
    assert runtime.bf16_reduced_precision_reduction == (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    with (arguments.capture_directory / "steps.jsonl").open() as steps:
        metadata = next(
            step
            for line in steps
            if (step := ProjectionStep.model_validate_json(line)).tensor_file
            == arguments.tensor_file
        )
    layout = metadata.full_projection_layouts[arguments.boundary]
    snapshots = torch.load(
        arguments.capture_directory / arguments.tensor_file,
        map_location="cpu",
        weights_only=True,
    )["full_projections"]
    captured_weights = torch.load(
        arguments.capture_directory / "actual-projection-weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    inputs = restore_matrix(
        snapshots[f"{arguments.boundary}.input"],
        [layout.row_stride_elements, layout.column_stride_elements],
        layout.storage_offset_elements,
    )
    weight = restore_matrix(
        captured_weights[arguments.boundary],
        runtime.weight_strides[arguments.boundary],
        runtime.weight_storage_offsets[arguments.boundary],
    )
    grouped_reference = F.linear(inputs, weight)
    assert torch.equal(
        grouped_reference.cpu(), snapshots[f"{arguments.boundary}.output"]
    )
    selected_input = inputs[:1]
    single_reference = F.linear(selected_input, weight)
    grouped_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(grouped_graph):
        grouped_output = F.linear(inputs, weight)
    single_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single_graph):
        single_output = F.linear(selected_input, weight)
    grouped_graph.replay()
    single_graph.replay()
    assert torch.equal(grouped_reference, grouped_output)
    assert torch.equal(single_reference, single_output)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("captured_grouped_projection")
    grouped_graph.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("same_address_single_row_projection")
    single_graph.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    assert torch.equal(grouped_reference, grouped_output)
    assert torch.equal(single_reference, single_output)
    report = {
        "tensor_file": arguments.tensor_file,
        "boundary": arguments.boundary,
        "phase": metadata.phase,
        "physical_rows": layout.rows,
        "input_shape": list(inputs.shape),
        "input_stride": list(inputs.stride()),
        "selected_row": 0,
        "grouped_input_pointer_bytes": inputs.data_ptr(),
        "single_input_pointer_bytes": selected_input.data_ptr(),
        "weight_shape": list(weight.shape),
        "row_changed_coordinates": int((grouped_output[:1] != single_output).sum()),
        "native_and_graph_outputs_exact": True,
        "runtime": runtime.model_dump(),
        "scope": "Actual captured geometry versus its first actual row at the same address. Profiling is excluded from serving performance claims.",
    }
    arguments.output_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
