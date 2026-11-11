"""Audit complete native decode lifetimes from their actual layer operands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    apply_rotary_pos_emb,
)

from scripts.debug_qwen3_omni_feedback_audit import TensorDifference, compare_tensors
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config
    register_checkpoint_conversion_mapping(
        configuration.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=configuration,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"]
    reference.eval()
    audit_checkpoint_weights(
        reference,
        Path(
            hf_hub_download(
                arguments.model_path, "config.json", revision=arguments.model_revision
            )
        ).parent,
        arguments.output_directory / "weight-audit.json",
    )
    text_configuration = configuration.text_config
    query_heads = text_configuration.num_attention_heads
    key_heads = text_configuration.num_key_value_heads
    head_dimensions = text_configuration.head_dim
    required_fields = {
        "positions",
        "input_contribution",
        "attention_normalized",
        "attention_residual",
        "qkv",
        "normalized_query",
        "normalized_key",
        "rotated_query",
        "rotated_key",
        "projected_value",
        "attention_values",
        "attention_output",
        "mlp_normalized",
        "mlp_residual",
        "mlp_output",
        "shared_output",
        "shared_gate",
        "router_logits",
        "expert_indices",
        "expert_weights",
        "routed_output",
    }
    reports = []
    for directory in sorted(arguments.capture_directory.iterdir()):
        finished = torch.load(directory / "finished.pt", weights_only=True)
        initial = torch.load(directory / "0000-output.pt", weights_only=True)
        assert bool(initial["past_exact"])
        assert (
            len(initial["keys"])
            == len(initial["values"])
            == len(reference.model.layers)
        )
        native_keys = [tensor.to(reference.device) for tensor in initial["keys"]]
        native_values = [tensor.to(reference.device) for tensor in initial["values"]]
        files = sorted(directory.glob("*-operators.pt"))
        assert [filename.name for filename in files] == [
            f"{frame:04d}-operators.pt"
            for frame in range(1, len(finished["output_ids"]))
        ]
        destination = arguments.output_directory / directory.name
        destination.mkdir()
        exact_counts: dict[str, int] = {}
        maximum_errors: dict[str, float] = {}
        layer_rows = 0
        for filename in files:
            frame = int(filename.name.split("-")[0])
            observed = torch.load(
                directory / f"{frame:04d}-output.pt", weights_only=True
            )
            before = torch.load(
                directory / f"{frame:04d}-before_decode.pt", weights_only=True
            )
            snapshots = torch.load(filename, weights_only=True)
            assert bool(observed["past_exact"])
            assert (
                len(observed["keys"])
                == len(observed["values"])
                == len(reference.model.layers)
            )
            assert len(snapshots) == len(reference.model.layers)
            assert int(observed["codec_ids"][0]) == int(finished["output_ids"][frame])
            assert torch.equal(
                snapshots[0]["input_contribution"],
                finished["decode_input_history"][frame - 1],
            )
            cosine, sine = reference.model.rotary_emb(
                snapshots[0]["input_contribution"].to(reference.device).view(1, 1, -1),
                before["positions"].to(reference.device).view(1, 1, 1).expand(3, 1, 1),
            )
            for index, (snapshot, layer) in enumerate(
                zip(snapshots, reference.model.layers, strict=True)
            ):
                assert set(snapshot) == required_fields | (
                    {"previous_residual"} if index else set()
                )
                assert snapshot["positions"].numel() in (1, 3)
                assert (snapshot["positions"] == before["positions"][-1]).all()
                for name, tensor in snapshot.items():
                    assert torch.isfinite(tensor).all()
                    if name == "positions":
                        assert tensor.dtype == torch.int64
                    elif name == "expert_indices":
                        assert tensor.dtype == torch.int32
                    elif name == "expert_weights":
                        assert tensor.dtype == torch.float32
                    else:
                        assert tensor.dtype == torch.bfloat16
                if index:
                    assert torch.equal(
                        snapshot["input_contribution"],
                        snapshots[index - 1]["mlp_output"],
                    )
                    assert torch.equal(
                        snapshot["previous_residual"],
                        snapshots[index - 1]["mlp_residual"],
                    )
                else:
                    pass
                tensors = {
                    name: tensor.to(reference.device).unsqueeze(0)
                    for name, tensor in snapshot.items()
                }
                differences: dict[str, TensorDifference] = {}
                for phase, contribution, previous, norm in (
                    (
                        "attention",
                        tensors["input_contribution"],
                        tensors.get("previous_residual"),
                        layer.input_layernorm,
                    ),
                    (
                        "mlp",
                        tensors["attention_output"],
                        tensors["attention_residual"],
                        layer.post_attention_layernorm,
                    ),
                ):
                    if previous is None:
                        normalized = rmsnorm(
                            contribution, norm.weight, norm.variance_epsilon
                        )
                        residual = contribution
                    else:
                        normalized, residual = contribution.clone(), previous.clone()
                        fused_add_rmsnorm(
                            normalized, residual, norm.weight, norm.variance_epsilon
                        )
                    assert torch.equal(residual, tensors[f"{phase}_residual"])
                    differences[f"{phase}_native_norm_replay"] = compare_tensors(
                        tensors[f"{phase}_normalized"], normalized
                    )
                    differences[f"{phase}_checkpoint_norm"] = compare_tensors(
                        tensors[f"{phase}_normalized"], norm(residual)
                    )
                projected = torch.nn.functional.linear(
                    tensors["attention_normalized"],
                    torch.cat(
                        (
                            layer.self_attn.q_proj.weight,
                            layer.self_attn.k_proj.weight,
                            layer.self_attn.v_proj.weight,
                        )
                    ),
                )
                differences["qkv_bfloat16_replay"] = compare_tensors(
                    tensors["qkv"], projected
                )
                query, key, value = tensors["qkv"].split(
                    (
                        query_heads * head_dimensions,
                        key_heads * head_dimensions,
                        key_heads * head_dimensions,
                    ),
                    dim=-1,
                )
                for name, raw, norm in (
                    ("query", query, layer.self_attn.q_norm),
                    ("key", key, layer.self_attn.k_norm),
                ):
                    normalized = rmsnorm(
                        raw.reshape(-1, head_dimensions),
                        norm.weight,
                        norm.variance_epsilon,
                    ).reshape_as(raw)
                    differences[f"{name}_standalone_norm_replay"] = compare_tensors(
                        tensors[f"normalized_{name}"], normalized
                    )
                    differences[f"{name}_checkpoint_norm"] = compare_tensors(
                        tensors[f"normalized_{name}"],
                        norm(raw.reshape(1, -1, head_dimensions)).reshape_as(raw),
                    )
                query_rotary, key_rotary = apply_rotary_pos_emb(
                    tensors["normalized_query"].reshape(
                        1, query_heads, 1, head_dimensions
                    ),
                    tensors["normalized_key"].reshape(1, key_heads, 1, head_dimensions),
                    cosine,
                    sine,
                )
                differences["query_checkpoint_rotary"] = compare_tensors(
                    tensors["rotated_query"], query_rotary.reshape_as(query)
                )
                differences["key_checkpoint_rotary"] = compare_tensors(
                    tensors["rotated_key"], key_rotary.reshape_as(key)
                )
                current_key = observed["keys"][index].to(reference.device)
                current_value = observed["values"][index].to(reference.device)
                assert (
                    current_key.shape
                    == current_value.shape
                    == (1, key_heads, head_dimensions)
                )
                assert current_key.dtype == current_value.dtype == torch.bfloat16
                assert (
                    native_keys[index].shape
                    == native_values[index].shape
                    == (int(before["positions"][-1]), key_heads, head_dimensions)
                )
                assert (
                    native_keys[index].dtype
                    == native_values[index].dtype
                    == torch.bfloat16
                )
                assert torch.equal(
                    tensors["rotated_key"].reshape_as(current_key), current_key
                )
                assert torch.equal(
                    tensors["projected_value"].reshape_as(current_value), current_value
                )
                assert torch.equal(value.reshape_as(current_value), current_value)
                native_keys[index] = torch.cat((native_keys[index], current_key))
                native_values[index] = torch.cat((native_values[index], current_value))
                assert len(native_keys[index]) == int(observed["sequence_length"])
                keys = (
                    native_keys[index]
                    .transpose(0, 1)[None]
                    .double()
                    .repeat_interleave(query_heads // key_heads, dim=1)
                )
                values = (
                    native_values[index]
                    .transpose(0, 1)[None]
                    .double()
                    .repeat_interleave(query_heads // key_heads, dim=1)
                )
                queries = (
                    tensors["rotated_query"]
                    .reshape(1, query_heads, 1, head_dimensions)
                    .double()
                )
                attention = (
                    torch.softmax(
                        (queries @ keys.transpose(-1, -2)) * head_dimensions**-0.5,
                        dim=-1,
                    )
                    @ values
                )
                differences["attention_float64_rounded"] = compare_tensors(
                    tensors["attention_values"],
                    attention.reshape(1, -1).to(reference.dtype),
                )
                differences["attention_projection_replay"] = compare_tensors(
                    tensors["attention_output"],
                    layer.self_attn.o_proj(tensors["attention_values"]),
                )
                logits = tensors["router_logits"]
                selected = tensors["expert_indices"].long()
                assert len(selected[0].unique()) == selected.shape[1]
                assert ((selected >= 0) & (selected < logits.shape[1])).all()
                assert (
                    logits.gather(1, selected)
                    >= logits.topk(selected.shape[1], dim=-1).values[:, -1:]
                ).all()
                weights = logits.float().softmax(-1).gather(1, selected)
                if text_configuration.norm_topk_prob:
                    weights /= weights.sum(-1, keepdim=True)
                else:
                    pass
                differences["router_projection_replay"] = compare_tensors(
                    logits,
                    torch.nn.functional.linear(
                        tensors["mlp_normalized"], layer.mlp.gate.weight
                    ),
                )
                differences["routing_weights_float32"] = compare_tensors(
                    tensors["expert_weights"], weights
                )
                differences["routing_weights_checkpoint_rounded"] = compare_tensors(
                    tensors["expert_weights"], weights.to(reference.dtype).float()
                )
                differences["shared_gate_projection_replay"] = compare_tensors(
                    tensors["shared_gate"],
                    layer.mlp.shared_expert_gate(tensors["mlp_normalized"]),
                )
                differences["shared_checkpoint_mlp"] = compare_tensors(
                    tensors["shared_output"],
                    layer.mlp.shared_expert(tensors["mlp_normalized"])
                    * torch.sigmoid(tensors["shared_gate"]),
                )
                checkpoint_routed = layer.mlp.experts(
                    tensors["mlp_normalized"], selected, tensors["expert_weights"]
                )
                differences["routed_checkpoint_mlp_actual_weights"] = compare_tensors(
                    tensors["routed_output"], checkpoint_routed
                )
                assert torch.equal(
                    tensors["routed_output"] + tensors["shared_output"],
                    tensors["mlp_output"],
                )
                if index == len(snapshots) - 1:
                    final_hidden = tensors["mlp_output"].clone()
                    final_residual = tensors["mlp_residual"].clone()
                    fused_add_rmsnorm(
                        final_hidden,
                        final_residual,
                        reference.model.norm.weight,
                        reference.model.norm.variance_epsilon,
                    )
                    differences["final_native_norm_replay"] = compare_tensors(
                        observed["hidden_states"].to(reference.device).unsqueeze(0),
                        final_hidden,
                    )
                    differences["head_reprojection"] = compare_tensors(
                        observed["logits"].to(reference.device).unsqueeze(0),
                        reference.codec_head(
                            observed["hidden_states"].to(reference.device).unsqueeze(0)
                        ).float(),
                    )
                else:
                    pass
                for name, difference in differences.items():
                    exact_counts[name] = exact_counts.get(name, 0) + difference.exact
                    maximum_errors[name] = max(
                        maximum_errors.get(name, 0), difference.maximum_absolute_error
                    )
                layer_rows += 1
                with (destination / "layers.jsonl").open("a") as output:
                    output.write(
                        json.dumps(
                            {
                                "frame": frame,
                                "layer": index,
                                "differences": {
                                    name: asdict(difference)
                                    for name, difference in differences.items()
                                },
                            }
                        )
                        + "\n"
                    )
            print(
                json.dumps(
                    {
                        "request_id": directory.name,
                        "frame": frame,
                        "layers": len(snapshots),
                    }
                ),
                flush=True,
            )
        report = {
            "request_id": directory.name,
            "decode_frames": len(files),
            "layer_rows": layer_rows,
            "exact_counts": exact_counts,
            "maximum_errors": maximum_errors,
        }
        reports.append(report)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(reports, indent=2)
        )


if __name__ == "__main__":
    main()
else:
    pass
