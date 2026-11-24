# SPDX-License-Identifier: Apache-2.0
"""Compare checkpoint operations at recorded native layer inputs."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from statistics import median

import torch
from audit_qwen3_omni_reference_backbone import TensorDifference, difference
from compare_qwen3_omni_native_trajectories import validate
from qwen3_omni_checkpoint_weights import load_talker
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    apply_rotary_pos_emb,
)


@dataclass(kw_only=True)
class BoundaryComparison:
    frame_index: int
    layer_index: int
    boundary: str
    reference_boundary: str
    selected_request: TensorDifference
    live_rows: TensorDifference


def checkpoint_operations(
    model: Qwen3OmniMoeTalkerForConditionalGeneration,
    layer_index: int,
    operands: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    layer = model.model.layers[layer_index]
    contribution = operands["contribution"]
    rows = contribution.shape[0]
    if layer_index == 0:
        attention_residual = contribution
    else:
        attention_residual = contribution + operands["previous_residual"]
    results = {
        "attention_residual": attention_residual,
        "attention_normalized": layer.input_layernorm(attention_residual),
    }
    attention = layer.self_attn
    widths = (
        attention.q_proj.out_features,
        attention.k_proj.out_features,
        attention.v_proj.out_features,
    )
    results["qkv_projection"] = torch.cat(
        [
            projection(operands["attention_normalized"])
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
        ],
        dim=-1,
    )
    query, key = operands["qkv_projection"].split(widths, dim=-1)[:2]
    results["query_before_rope"] = attention.q_norm(
        query.reshape(rows, 1, -1, attention.head_dim)
    ).reshape(rows, -1)
    results["key_before_rope"] = attention.k_norm(
        key.reshape(rows, 1, -1, attention.head_dim)
    ).reshape(rows, -1)
    positions = operands["positions"].transpose(0, 1).unsqueeze(-1)
    assert positions.shape == (3, rows, 1)
    cosine, sine = model.model.rotary_emb(contribution, positions)
    query, key = apply_rotary_pos_emb(
        operands["query_before_rope"].reshape(rows, -1, 1, attention.head_dim),
        operands["key_before_rope"].reshape(rows, -1, 1, attention.head_dim),
        cosine,
        sine,
    )
    results["query_after_rope"] = query.reshape(rows, -1)
    results["key_after_rope"] = key.reshape(rows, -1)
    results["attention_projection"] = attention.o_proj(operands["attention_context"])
    results["mlp_residual"] = (
        operands["attention_residual"] + operands["attention_projection"]
    )
    results["mlp_normalized"] = layer.post_attention_layernorm(results["mlp_residual"])
    normalized = operands["mlp_normalized"]
    results["router_logits"], checkpoint_weights, checkpoint_ids = layer.mlp.gate(
        normalized
    )
    results["shared_expert"] = layer.mlp.shared_expert(normalized)
    results["shared_gate"] = layer.mlp.shared_expert_gate(normalized)
    results["routed_experts"] = layer.mlp.experts(
        normalized, operands["expert_ids"].long(), operands["expert_weights"]
    )
    results["routed_experts_rounded_weights"] = layer.mlp.experts(
        normalized,
        operands["expert_ids"].long(),
        operands["expert_weights"].to(normalized.dtype),
    )
    results["routed_experts_checkpoint_routing"] = layer.mlp.experts(
        normalized, checkpoint_ids, checkpoint_weights
    )
    results["expert_weight_cast_effect"] = results["routed_experts_rounded_weights"]
    results["layer_output"] = results["routed_experts_checkpoint_routing"] + (
        results["shared_gate"].sigmoid() * results["shared_expert"]
    )
    if layer_index == len(model.model.layers) - 1:
        results["predictor_query"] = model.model.norm(
            operands["layer_output"] + operands["mlp_residual"]
        )
    else:
        pass
    return results


@torch.inference_mode()
def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    torch.set_num_threads(1)
    torch.cuda.set_device(arguments.gpu)
    model = load_talker(arguments.checkpoint, arguments.gpu, arguments.output)
    native_receipt = arguments.root / "native-routing-boundaries/boundaries.json"
    native_controls = json.loads(native_receipt.read_text())["captures"]
    captures = sorted(arguments.root.glob("captures/*/trajectory.pt"))
    assert captures
    for capture in captures:
        capture_hash = sha256(capture.read_bytes()).hexdigest()
        native_control = native_controls[capture.parent.name]
        assert native_control["capture_sha256"] == capture_hash
        assert not native_control["mismatches"]
        frames = validate(capture.parent)
        metadata = json.loads(capture.with_suffix(".json").read_text())["frames"]
        comparisons: list[BoundaryComparison] = []
        for frame_index, (frame, step) in enumerate(zip(frames, metadata, strict=True)):
            if "captured_layers" not in step:
                continue
            else:
                assert step["captured_layers"] == len(model.model.layers)
            live_rows = len(step["request_ids"])
            selected_row = step["request_row"]
            for layer_index in range(len(model.model.layers)):
                prefix = f"layer{layer_index}_peer_"
                operands = {
                    name.removeprefix(prefix): tensor.to(model.device)
                    for name, tensor in frame.items()
                    if name.startswith(prefix)
                }
                assert operands["contribution"].shape[0] == step["graph_batch_rows"]
                outputs = checkpoint_operations(model, layer_index, operands)
                repeated_outputs = checkpoint_operations(model, layer_index, operands)
                assert outputs.keys() == repeated_outputs.keys()
                for boundary, output in outputs.items():
                    assert torch.equal(output, repeated_outputs[boundary]), (
                        frame_index,
                        layer_index,
                        boundary,
                    )
                    if boundary == "predictor_query":
                        expected = (
                            frame["predictor_query"].reshape(1, -1).to(model.device)
                        )
                        output = output[selected_row : selected_row + 1]
                        comparison_row = 0
                        reference_boundary = boundary
                    elif boundary == "expert_weight_cast_effect":
                        reference_boundary = (
                            "checkpoint_experts_with_native_fp32_weights"
                        )
                        expected = outputs["routed_experts"][:live_rows]
                        output = output[:live_rows]
                        comparison_row = selected_row
                    else:
                        reference_boundary = (
                            "routed_experts"
                            if boundary.startswith("routed_experts_")
                            else boundary
                        )
                        expected = operands[reference_boundary][:live_rows]
                        output = output[:live_rows]
                        comparison_row = selected_row
                    comparisons.append(
                        BoundaryComparison(
                            frame_index=frame_index,
                            layer_index=layer_index,
                            boundary=boundary,
                            reference_boundary=reference_boundary,
                            selected_request=difference(
                                expected[comparison_row], output[comparison_row]
                            ),
                            live_rows=difference(expected, output),
                        )
                    )
                for name, operand in operands.items():
                    assert torch.equal(operand.cpu(), frame[prefix + name]), (
                        frame_index,
                        layer_index,
                        name,
                    )
            print(
                f"{capture.parent.name}: completed deep frame {frame_index}", flush=True
            )
        assert comparisons
        summaries = {}
        for boundary in sorted({row.boundary for row in comparisons}):
            selected = [
                row.selected_request for row in comparisons if row.boundary == boundary
            ]
            relative_errors = [
                row.relative_l2 for row in selected if row.relative_l2 is not None
            ]
            summaries[boundary] = {
                "compared_instances": len(selected),
                "exact_selected_request_instances": sum(
                    row.changed_values == 0 for row in selected
                ),
                "median_selected_relative_l2": (
                    median(relative_errors) if relative_errors else None
                ),
                "maximum_selected_relative_l2": (
                    max(relative_errors) if relative_errors else None
                ),
                "maximum_selected_absolute_error": max(
                    row.maximum_absolute for row in selected
                ),
            }
        report = {
            "scope": "Checkpoint eager BF16 operations at recorded native inputs. Peer projection geometry is preserved. Routed experts separately hold native IDs/FP32 scales, cast those scales to BF16, or recompute checkpoint routing. Independent operators do not consume preceding checkpoint outputs, except each residual-add/norm pair and the explicitly combined MLP. Attention context is held native; it has a separate FP64 audit. Not a free-running speech or performance result.",
            "capture_sha256": capture_hash,
            "native_control_sha256": sha256(native_receipt.read_bytes()).hexdigest(),
            "repeated_outputs_exact": True,
            "inputs_unchanged": True,
            "summaries": summaries,
            "comparisons": [asdict(row) for row in comparisons],
        }
        with (arguments.output / f"{capture.parent.name}.json").open("x") as receipt:
            json.dump(report, receipt, indent=2)
        print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    main(parser.parse_args())
else:
    pass
