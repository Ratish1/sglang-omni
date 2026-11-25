# SPDX-License-Identifier: Apache-2.0
"""Check masked normalization against held native and checkpoint operands."""

import argparse
import json
from hashlib import sha256
from os import environ
from pathlib import Path

import torch
from audit_qwen3_omni_native_boundaries import verify_weights
from audit_qwen3_omni_native_predictor import load_native_talker
from compare_qwen3_omni_native_trajectories import validate
from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRMSNorm,
)


@torch.inference_mode()
def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    environ["OMNI_NATIVE_TRAJECTORY_DIRECTORY"] = str(arguments.output)
    environ["OMNI_NATIVE_TRAJECTORY_SAMPLES"] = arguments.capture.name
    assert not environ.get("OMNI_NATIVE_QK_TRIAL")
    assert arguments.conditioning.name == arguments.capture.name
    environ["OMNI_NATIVE_QK_TRIAL"] = "control"
    environ["OMNI_NATIVE_CONDITIONING_REFERENCES"] = json.dumps(
        [str(arguments.conditioning)]
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(arguments.gpu)
    scheduler = load_native_talker(
        arguments.checkpoint, arguments.config, arguments.gpu, arguments.output
    )
    model = scheduler.model_runner.model
    verified_weights = verify_weights(model, arguments.checkpoint)
    frames = validate(arguments.capture)
    metadata = json.loads((arguments.capture / "trajectory.json").read_text())
    assert model.native_qk_trial is not None
    selected_rows = model.native_qk_trial.selected_rows
    trials = model.native_qk_trial.layers
    comparisons: list[dict[str, int]] = []
    kernel_comparisons: list[dict[str, int | float | str]] = []
    for frame_index, (frame, state) in enumerate(
        zip(frames, metadata["frames"], strict=True)
    ):
        if "captured_layers" not in state:
            continue
        else:
            assert state["captured_layers"] == len(model.model.layers)
        row = state["request_row"]
        for layer_index, layer in enumerate(model.model.layers):
            prefix = f"layer{layer_index}_peer_"
            projection = frame[prefix + "qkv_projection"].cuda()
            query = frame[prefix + "query_before_rope"].cuda()
            key = frame[prefix + "key_before_rope"].cuda()
            positions = frame[prefix + "positions"].cuda().movedim(0, -1)
            original = (projection.clone(), query.clone(), key.clone())
            mask = selected_rows[: len(query)]
            trial = trials[layer_index]
            trial.is_decode = True
            trial.projection = projection
            inputs = (positions, query, key)
            for repeat in range(2):
                result = trial.before_rotary(layer.self_attn.rotary_emb, inputs)
                assert result is not None
                assert torch.equal(result[1], query) and torch.equal(result[2], key)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = trial.before_rotary(layer.self_attn.rotary_emb, inputs)
            assert result is not None
            raw_operands = projection.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )[:2]
            expected: list[torch.Tensor] = []
            for operand_name, raw, normalization in zip(
                ("query", "key"),
                raw_operands,
                (layer.self_attn.q_norm, layer.self_attn.k_norm),
                strict=True,
            ):
                reference = Qwen3OmniMoeThinkerTextRMSNorm(
                    layer.self_attn.head_dim, normalization.variance_epsilon
                ).cuda()
                reference.weight = normalization.weight
                expected.append(
                    reference(raw.reshape(len(raw), -1, layer.self_attn.head_dim))
                    .reshape_as(raw)
                    .clone()
                )
                kernel_input = raw.reshape(-1, layer.self_attn.head_dim).contiguous()
                kernel_output = rmsnorm_hf(
                    kernel_input, normalization.weight, normalization.variance_epsilon
                )
                kernel_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(kernel_graph):
                    captured_kernel_output = rmsnorm_hf(
                        kernel_input,
                        normalization.weight,
                        normalization.variance_epsilon,
                    )
                kernel_graph.replay()
                assert torch.equal(captured_kernel_output, kernel_output)
                kernel_output = kernel_output.reshape_as(raw)
                selected_kernel_output = rmsnorm_hf(
                    raw[row].reshape(-1, layer.self_attn.head_dim).contiguous(),
                    normalization.weight,
                    normalization.variance_epsilon,
                ).reshape_as(raw[row])
                assert torch.equal(selected_kernel_output, kernel_output[row])
                live_rows = len(state["request_ids"])
                assert torch.isfinite(kernel_output[:live_rows]).all()
                changed_values = kernel_output != expected[-1]
                absolute_difference = (
                    kernel_output.float() - expected[-1].float()
                ).abs()
                kernel_comparisons.append(
                    {
                        "frame": frame_index,
                        "layer": layer_index,
                        "operand": operand_name,
                        "live_rows": live_rows,
                        "selected_changed_values": int(changed_values[row].sum()),
                        "live_changed_values": int(changed_values[:live_rows].sum()),
                        "selected_max_absolute_difference": float(
                            absolute_difference[row].max()
                        ),
                        "live_max_absolute_difference": float(
                            absolute_difference[:live_rows].max()
                        ),
                    }
                )
            # note (ratish): moving the mask tests a graph buffer, not a capture-time Python choice.
            for selected_row in (row, (row + 1) % len(state["request_ids"]), row):
                selected_rows.zero_()
                selected_rows[selected_row] = True
                graph.replay()
                for actual, native, checkpoint in zip(
                    result[1:], (query, key), expected, strict=True
                ):
                    assert torch.equal(actual[selected_row], checkpoint[selected_row])
                    assert torch.equal(actual[~mask], native[~mask])
                selected_rows.zero_()
                graph.replay()
                assert torch.equal(result[1], query) and torch.equal(result[2], key)
            for actual, before in zip((projection, query, key), original, strict=True):
                assert torch.equal(actual, before)
            trial.finish(layer.self_attn, (), frame[prefix + "attention_projection"])
            assert trial.before_rotary(layer.self_attn.rotary_emb, inputs) is None
            assert trial.projection is None
            comparisons.append(
                {"frame": frame_index, "layer": layer_index, "batch_rows": len(query)}
            )
    assert comparisons
    report = {
        "capture_sha256": sha256(
            (arguments.capture / "trajectory.pt").read_bytes()
        ).hexdigest(),
        "verified_weights": len(verified_weights),
        "checked_layer_frames": len(comparisons),
        "distinct_rotary_modules": len(
            {id(layer.self_attn.rotary_emb) for layer in model.model.layers}
        ),
        "checks": {
            "control_exact": True,
            "checkpoint_selected_row_exact": True,
            "unselected_rows_unchanged": True,
            "graph_mask_moves_and_resets": True,
            "operands_unchanged": True,
            "inactive_after_attention_returns": True,
            "native_factory_control_graph_capture": True,
            "hf_kernel_graph_matches_eager": True,
            "hf_kernel_selected_row_matches_batched": True,
        },
        "scope": "Held Q/K input preflight only; native serving control still required.",
        "comparisons": comparisons,
        "hf_kernel_comparisons": kernel_comparisons,
    }
    (arguments.output / "qk-trial.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: value
                for name, value in report.items()
                if name not in ("comparisons", "hf_kernel_comparisons")
            }
        )
    )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    main(parser.parse_args())
else:
    pass
