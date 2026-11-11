"""Replay complete native GEMMs while isolating the BF16 reduction option."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from safetensors import safe_open

from scripts.debug_qwen3_omni_compare_thinker_decisions import CapturedStep
from scripts.debug_qwen3_omni_thinker_decisions import CheckpointIndex


class ProjectionLayout(BaseModel):
    rows: int
    row_stride_elements: int
    column_stride_elements: int
    storage_offset_elements: int
    data_pointer_bytes: int


class ProjectionStep(CapturedStep):
    full_projection_layouts: dict[str, ProjectionLayout]


class ProjectionRuntime(BaseModel):
    bf16_gemm_backend: str
    bf16_reduced_precision_reduction: bool
    preferred_blas_library: str
    quantization_methods: dict[str, str]
    weight_strides: dict[str, list[int]]
    weight_storage_offsets: dict[str, int]


@dataclass(kw_only=True)
class ProjectionReplay:
    eager: torch.Tensor
    graph: torch.Tensor
    repeat_exact: bool


def restore_matrix(
    tensor: torch.Tensor, strides: list[int], storage_offset_elements: int
) -> torch.Tensor:
    storage_elements = (
        storage_offset_elements
        + sum(
            (length - 1) * stride
            for length, stride in zip(tensor.shape, strides, strict=True)
        )
        + 1
    )
    storage = torch.empty(storage_elements, dtype=tensor.dtype, device="cuda")
    restored = storage.as_strided(tensor.shape, strides, storage_offset_elements)
    restored.copy_(tensor)
    return restored


def replay_projection(inputs: torch.Tensor, weight: torch.Tensor) -> ProjectionReplay:
    eager = F.linear(inputs, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = F.linear(inputs, weight)
    graph.replay()
    first = output.clone()
    graph.replay()
    return ProjectionReplay(
        eager=eager.cpu(), graph=first.cpu(), repeat_exact=torch.equal(first, output)
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    runtime = ProjectionRuntime.model_validate_json(
        (arguments.capture_directory / "projection-runtime.json").read_bytes()
    )
    assert runtime.bf16_gemm_backend in ("auto", "torch")
    assert set(runtime.quantization_methods.values()) == {"UnquantizedLinearMethod"}
    assert (
        str(torch.backends.cuda.preferred_blas_library())
        == runtime.preferred_blas_library
    )
    original_reduction = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    original_split_k = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k
    )
    assert original_reduction == runtime.bf16_reduced_precision_reduction
    index = CheckpointIndex.model_validate_json(
        (arguments.checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    captured_weights = torch.load(
        arguments.capture_directory / "actual-projection-weights.pt",
        weights_only=True,
        map_location="cpu",
    )
    weights: dict[str, torch.Tensor] = {}
    for name, tensor in captured_weights.items():
        layer_index = int(name.split(".")[1])
        weight_name = f"thinker.model.layers.{layer_index}.self_attn.o_proj.weight"
        with safe_open(
            arguments.checkpoint_directory / index.weight_map[weight_name],
            framework="pt",
            device="cpu",
        ) as checkpoint:
            assert torch.equal(tensor, checkpoint.get_tensor(weight_name))
        weights[name] = restore_matrix(
            tensor, runtime.weight_strides[name], runtime.weight_storage_offsets[name]
        )
    arguments.output_directory.mkdir(exist_ok=False)
    counts: Counter[str] = Counter()
    try:
        with (
            (arguments.capture_directory / "steps.jsonl").open() as steps,
            (arguments.output_directory / "replay.jsonl").open("w") as output,
        ):
            for line in steps:
                metadata = ProjectionStep.model_validate_json(line)
                snapshots = torch.load(
                    arguments.capture_directory / metadata.tensor_file,
                    map_location="cpu",
                    weights_only=True,
                )["full_projections"]
                assert set(metadata.full_projection_layouts) == set(weights)
                for name, layout in metadata.full_projection_layouts.items():
                    saved_input = snapshots[f"{name}.input"]
                    actual = snapshots[f"{name}.output"]
                    assert saved_input.shape[0] == actual.shape[0] == layout.rows
                    inputs = restore_matrix(
                        saved_input,
                        [layout.row_stride_elements, layout.column_stride_elements],
                        layout.storage_offset_elements,
                    )
                    weight = weights[name]
                    expected = (inputs.double() @ weight.double().T).cpu()
                    assert bool(expected.isfinite().all() & actual.isfinite().all())
                    observations: dict[str, ProjectionReplay] = {}
                    measurements: dict[str, dict[str, bool | int | float]] = {}
                    for label, allow_reduction in (
                        ("default_before", original_reduction),
                        ("no_reduced_precision", False),
                        ("default_after", original_reduction),
                    ):
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                            allow_reduction,
                            original_split_k,
                        )
                        observed = replay_projection(inputs, weight)
                        observations[label] = observed
                        error = observed.graph.double() - expected
                        measurements[label] = {
                            "eager_matches_native": torch.equal(observed.eager, actual),
                            "graph_matches_native": torch.equal(observed.graph, actual),
                            "eager_equals_graph": torch.equal(
                                observed.eager, observed.graph
                            ),
                            "graph_repeat_exact": observed.repeat_exact,
                            "changed_native_coordinates": int(
                                (observed.graph != actual).sum()
                            ),
                            "maximum_absolute_error_to_fp64": float(error.abs().max()),
                        }
                    before = observations["default_before"].graph
                    disabled = observations["no_reduced_precision"].graph
                    after = observations["default_after"].graph
                    restored = torch.equal(before, after)
                    changed = before != disabled
                    counts["matrix_calls"] += 1
                    counts["physical_rows"] += layout.rows
                    counts["native_graph_exact"] += measurements["default_before"][
                        "graph_matches_native"
                    ]
                    counts["restored_default_exact"] += restored
                    counts["reduction_changes_output"] += bool(changed.any())
                    counts["changed_coordinates"] += int(changed.sum())
                    if bool(changed.any()):
                        selected_rows = changed.any(-1).nonzero().flatten()
                        torch.save(
                            {
                                "rows": selected_rows,
                                "before": before[selected_rows],
                                "disabled": disabled[selected_rows],
                                "after": after[selected_rows],
                            },
                            arguments.output_directory
                            / f"{metadata.tensor_file.removesuffix('.pt')}-{name}.pt",
                        )
                    else:
                        pass
                    output.write(
                        json.dumps(
                            {
                                "tensor_file": metadata.tensor_file,
                                "phase": metadata.phase,
                                "boundary": name,
                                "shape": list(inputs.shape),
                                "original_layout": layout.model_dump(),
                                "replay_data_pointer_bytes": inputs.data_ptr(),
                                "restored_default_exact": restored,
                                "reduction_changed_coordinates": int(changed.sum()),
                                "measurements": measurements,
                            }
                        )
                        + "\n"
                    )
                output.flush()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            original_reduction,
            original_split_k,
        )
    report = {
        "counts": dict(counts),
        "runtime": runtime.model_dump(),
        "scope": "All actual padded matrices, original strides/offsets and checkpoint-equal loaded weights. No synthetic rows and no model generation under changed precision.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
