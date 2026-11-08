"""Verify expert references and replay native MoE only with complete original rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerTextExperts,
)

from scripts.debug_qwen3_omni_expert_corpus import (
    compare_outputs,
    load_layer_weights,
    reference_mlp,
)


class ReplayRow(BaseModel):
    request_id: str
    frame: int
    step: int
    source_row: int
    live_rows: int
    projection_rows: int
    phase: str


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expert-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--tune-max-num-tokens", type=int, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config.text_config
    model_directory = Path(
        snapshot_download(
            arguments.model_path,
            revision=arguments.model_revision,
            local_files_only=True,
        )
    )
    weight_map = json.loads(
        (model_directory / "model.safetensors.index.json").read_text()
    )["weight_map"]
    packed = torch.load(
        arguments.expert_directory / "selected-routes.pt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    metadata = [
        ReplayRow.model_validate_json(line)
        for line in (arguments.expert_directory / "rows.jsonl").read_text().splitlines()
    ]
    assert len(metadata) == packed["router.input"].shape[1]
    steps: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        steps[row.step].append(index)
    eligible = {
        step: indices
        for step, indices in steps.items()
        if len(indices) == metadata[indices[0]].projection_rows
    }
    for indices in eligible.values():
        assert [metadata[index].source_row for index in indices] == list(
            range(len(indices))
        )
        assert all(metadata[index].live_rows == len(indices) for index in indices)
    assert eligible
    eligible_rows = sorted(index for indices in eligible.values() for index in indices)
    summary = {
        "model_revision": arguments.model_revision,
        "frames": len(metadata),
        "eligible_frames": len(eligible_rows),
        "eligible_steps": len(eligible),
        "skipped_steps": [
            {
                "step": step,
                "phase": metadata[indices[0]].phase,
                "live_rows": len(indices),
                "projection_rows": metadata[indices[0]].projection_rows,
            }
            for step, indices in steps.items()
            if step not in eligible
        ],
        "use_fused_finalize": False,
        "tune_max_num_tokens": arguments.tune_max_num_tokens,
        "layers": [],
        "scope": "Native CUTLASS replay uses all original rows, IDs and scales only where captured. Missing prefill/padding rows are never synthesized. Eager replay is measured against captured graph output, not assumed identical. Reference validation runs the independent checkpoint Experts forward in FP64 on the actual first live batch.",
    }
    with (arguments.output_directory / "replay.jsonl").open("w") as reports:
        for layer in range(config.num_hidden_layers):
            weights = load_layer_weights(
                model_directory, layer, config.num_experts, weight_map
            )
            first_rows = next(iter(eligible.values()))
            inputs = packed["router.input"][layer, first_rows].cuda().double()
            ids = packed["expert_ids"][layer, first_rows].cuda().long()
            scales = packed["expert_weights"][layer, first_rows].cuda().double()
            with torch.device("cuda"):
                checkpoint_experts = Qwen3OmniMoeTalkerTextExperts(config).double()
            expected = torch.zeros_like(inputs)
            for expert in range(config.num_experts):
                gate = weights[f"experts.{expert}.gate_proj.weight"]
                up = weights[f"experts.{expert}.up_proj.weight"]
                down = weights[f"experts.{expert}.down_proj.weight"]
                checkpoint_experts.gate_up_proj[expert].copy_(torch.cat((gate, up)))
                checkpoint_experts.down_proj[expert].copy_(down)
                rows, slots = torch.where(ids == expert)
                if rows.numel() > 0:
                    reference = reference_mlp(inputs[rows], gate, up, down).outputs[
                        "fp64"
                    ]
                    expected[rows] += reference * scales[rows, slots, None]
                else:
                    pass
            checkpoint_output = checkpoint_experts(inputs, ids, scales)
            torch.testing.assert_close(checkpoint_output, expected)
            checkpoint_maximum = float((checkpoint_output - expected).abs().max())
            w13 = torch.stack(
                [
                    torch.cat(
                        (
                            weights[f"experts.{expert}.up_proj.weight"],
                            weights[f"experts.{expert}.gate_proj.weight"],
                        )
                    )
                    for expert in range(config.num_experts)
                ]
            )
            w2 = torch.stack(
                [
                    weights[f"experts.{expert}.down_proj.weight"]
                    for expert in range(config.num_experts)
                ]
            )
            measurements = {
                name: torch.empty(len(metadata), dtype=torch.float64)
                for name in ("relative_l2", "max_absolute", "equal_coordinates")
            }
            outputs = []
            exact_steps = 0
            for ordinal, (step, indices) in enumerate(eligible.items()):
                inputs = packed["router.input"][layer, indices].cuda()
                ids = packed["expert_ids"][layer, indices].cuda().int()
                scales = packed["expert_weights"][layer, indices].cuda()
                actual = packed["routed"][layer, indices].cuda()
                output = cutlass_fused_moe(
                    input=inputs,
                    token_selected_experts=ids,
                    token_final_scales=scales,
                    fc1_expert_weights=w13,
                    fc2_expert_weights=w2,
                    output_dtype=inputs.dtype,
                    quant_scales=None,
                    use_fused_finalize=False,
                    tune_max_num_tokens=arguments.tune_max_num_tokens,
                )[0]
                metrics = compare_outputs(actual, output)
                assert metrics["finite"].all()
                exact = torch.equal(actual, output)
                exact_steps += int(exact)
                for name, values in measurements.items():
                    values[indices] = metrics[name].cpu()
                outputs.append(output.cpu())
                reports.write(
                    json.dumps(
                        {
                            "layer": layer,
                            "step": step,
                            "exact": exact,
                            "max_absolute": float(metrics["max_absolute"].max()),
                        }
                    )
                    + "\n"
                )
                if ordinal % 500 == 0:
                    print(
                        json.dumps(
                            {
                                "layer": layer,
                                "native_steps": ordinal + 1,
                                "exact_steps": exact_steps,
                            }
                        ),
                        flush=True,
                    )
                else:
                    pass
            row_indices = [index for indices in eligible.values() for index in indices]
            torch.save(
                {
                    "row_indices": row_indices,
                    "outputs": torch.cat(outputs),
                    "measurements": {
                        name: values[row_indices]
                        for name, values in measurements.items()
                    },
                },
                arguments.output_directory / f"layer-{layer:02d}.pt",
            )
            summary["layers"].append(
                {
                    "layer": layer,
                    "checkpoint_reference_verified": True,
                    "checkpoint_max_absolute": checkpoint_maximum,
                    "steps": len(eligible),
                    "exact_steps": exact_steps,
                }
            )
            reports.flush()
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(summary, indent=2)
            )


if __name__ == "__main__":
    main()
