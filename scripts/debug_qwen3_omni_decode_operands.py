"""Audit complete native decode lifetimes from their actual layer operands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
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
from scripts.debug_qwen3_omni_sampling_alignment import SamplingParameters
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--include-prefill", action="store_true")
    parser.add_argument("--full-prefill-prefix", action="store_true")
    arguments = parser.parse_args()
    assert not arguments.full_prefill_prefix or arguments.include_prefill
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
    directories = sorted(
        path for path in arguments.capture_directory.iterdir() if path.is_dir()
    )
    assert directories
    for directory in directories:
        sampling = SamplingParameters.model_validate_json(
            (directory / "sampling.json").read_bytes()
        )
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
            for frame in range(
                0 if arguments.include_prefill else 1, len(finished["output_ids"])
            )
        ]
        destination = arguments.output_directory / directory.name
        destination.mkdir()
        exact_counts: dict[str, int] = {}
        maximum_errors: dict[str, float] = {}
        layer_rows = 0
        layer_steps = 0
        for filename in files:
            frame = int(filename.name.split("-")[0])
            full_prefix = frame == 0 and arguments.full_prefill_prefix
            observed = torch.load(
                directory / f"{frame:04d}-output.pt", weights_only=True
            )
            phase = "before_prefill" if frame == 0 else "before_decode"
            before = torch.load(
                directory / f"{frame:04d}-{phase}.pt", weights_only=True
            )
            snapshots = torch.load(filename, weights_only=True)
            assert bool(observed["past_exact"])
            assert bool(observed["is_extend"]) == (frame == 0)
            assert (
                len(observed["keys"])
                == len(observed["values"])
                == len(reference.model.layers)
            )
            assert len(snapshots) == len(reference.model.layers)
            assert int(observed["codec_ids"][0]) == int(finished["output_ids"][frame])
            assert torch.equal(
                snapshots[0]["input_contribution"],
                (
                    before["prefill_embeds"][-1]
                    if frame == 0
                    else finished["decode_input_history"][frame - 1]
                ),
            )
            query_positions = (
                before["positions"] if full_prefix else before["positions"][-1:]
            )
            first_inputs = (
                snapshots[0]["prefix_input_contribution"]
                if full_prefix
                else snapshots[0]["input_contribution"].unsqueeze(0)
            )
            position_axes = (
                snapshots[0]["prefix_positions"].reshape(len(query_positions), -1)
                if full_prefix
                else snapshots[0]["positions"].reshape(1, -1)
            )
            assert position_axes.shape[1] in (1, 3)
            assert torch.equal(position_axes[:, 0], query_positions)
            if full_prefix:
                assert torch.equal(first_inputs, before["prefill_embeds"])
            else:
                pass
            if frame == 0:
                assert int(before["extend_prefix_length"]) == 0
                assert torch.equal(
                    before["positions"], torch.arange(len(before["prefill_embeds"]))
                )
                assert len(before["prefill_embeds"]) == int(observed["sequence_length"])
            else:
                pass
            cosine, sine = reference.model.rotary_emb(
                first_inputs.to(reference.device).unsqueeze(0),
                position_axes.to(reference.device)
                .T.unsqueeze(1)
                .expand(3, 1, len(query_positions)),
            )
            previous_snapshot = None
            for index, (snapshot, layer) in enumerate(
                zip(snapshots, reference.model.layers, strict=True)
            ):
                metadata_fields = set()
                expected_fields = required_fields | (
                    {"previous_residual"} if index else set()
                )
                if "batch_selected_row" in snapshot:
                    batch_fields = {
                        "mlp_normalized",
                        "router_logits",
                        "expert_indices",
                        "expert_weights",
                        "routed_output",
                    }
                    metadata_fields = (
                        {f"batch_{name}" for name in batch_fields}
                        | {f"layout_{name}" for name in batch_fields}
                        | {
                            "batch_selected_row",
                            "batch_is_graph",
                            "layout_qkv",
                            "layout_is_prefill",
                        }
                    )
                    assert bool(snapshot["layout_is_prefill"]) == (frame == 0)
                    packed_layout = snapshot["layout_qkv"].tolist()
                    assert packed_layout == [
                        len(snapshot["batch_mlp_normalized"]),
                        (query_heads + 2 * key_heads) * head_dimensions,
                        (query_heads + 2 * key_heads) * head_dimensions,
                        1,
                        0,
                    ]
                    if "prefix_token_rows" in snapshot:
                        metadata_fields.add("prefix_token_rows")
                        metadata_fields.update(
                            f"prefix_{name}" for name in expected_fields
                        )
                        start, stop = snapshot["prefix_token_rows"].tolist()
                        assert (
                            0 <= start < stop <= len(snapshot["batch_mlp_normalized"])
                        )
                        assert stop - start == len(before["positions"])
                        assert stop == int(snapshot["batch_selected_row"]) + 1
                        for name in expected_fields:
                            assert torch.equal(
                                snapshot[f"prefix_{name}"][-1], snapshot[name]
                            )
                        for name in batch_fields:
                            assert torch.equal(
                                snapshot[f"prefix_{name}"],
                                snapshot[f"batch_{name}"][start:stop],
                            )
                    elif frame == 0:
                        metadata_fields.add("layout_prefill_token_row")
                        assert int(snapshot["layout_prefill_token_row"][0]) == int(
                            snapshot["batch_selected_row"]
                        )
                    else:
                        pass
                    for name in batch_fields:
                        assert torch.equal(
                            snapshot[f"batch_{name}"][
                                int(snapshot["batch_selected_row"])
                            ],
                            snapshot[name],
                        )
                else:
                    pass
                assert set(snapshot) == expected_fields | metadata_fields
                assert not full_prefix or "prefix_token_rows" in metadata_fields
                snapshot = {
                    name: snapshot[f"prefix_{name}" if full_prefix else name]
                    for name in expected_fields
                }
                observed_axes = (
                    snapshot["positions"].reshape(len(query_positions), -1)
                    if full_prefix
                    else snapshot["positions"].reshape(1, -1)
                )
                assert torch.equal(observed_axes, position_axes)
                if (
                    frame == 0
                    and before["mrope_positions"].numel()
                    and observed_axes.shape[1] == 3
                ):
                    expected_axes = (
                        before["mrope_positions"].T
                        if full_prefix
                        else before["mrope_positions"][:, -1:].T
                    )
                    assert torch.equal(observed_axes, expected_axes)
                else:
                    pass
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
                    assert previous_snapshot is not None
                    assert torch.equal(
                        snapshot["input_contribution"],
                        previous_snapshot["mlp_output"],
                    )
                    assert torch.equal(
                        snapshot["previous_residual"],
                        previous_snapshot["mlp_residual"],
                    )
                else:
                    pass
                tensors = {
                    name: (
                        tensor.to(reference.device)
                        if full_prefix
                        else tensor.to(reference.device).unsqueeze(0)
                    )
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
                paired_projection = tensors["qkv"].clone()
                paired_query = paired_projection.as_strided(
                    (len(query_positions), query_heads, head_dimensions),
                    (paired_projection.stride(0), head_dimensions, 1),
                )
                paired_key = paired_projection.as_strided(
                    (len(query_positions), key_heads, head_dimensions),
                    (paired_projection.stride(0), head_dimensions, 1),
                    storage_offset=query_heads * head_dimensions,
                )
                assert (
                    paired_projection.stride(0)
                    == (query_heads + 2 * key_heads) * head_dimensions
                )
                assert all(
                    tensor.data_ptr() % 16 == 0
                    for tensor in (
                        paired_query,
                        paired_key,
                        layer.self_attn.q_norm.weight,
                        layer.self_attn.k_norm.weight,
                    )
                )
                fused_inplace_qknorm(
                    q=paired_query,
                    k=paired_key,
                    q_weight=layer.self_attn.q_norm.weight,
                    k_weight=layer.self_attn.k_norm.weight,
                    eps=layer.self_attn.q_norm.variance_epsilon,
                    head_dim=head_dimensions,
                )
                differences["query_paired_norm_replay"] = compare_tensors(
                    tensors["normalized_query"], paired_query.reshape_as(query)
                )
                differences["key_paired_norm_replay"] = compare_tensors(
                    tensors["normalized_key"], paired_key.reshape_as(key)
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
                    tensors["normalized_query"]
                    .reshape(1, len(query_positions), query_heads, head_dimensions)
                    .transpose(1, 2),
                    tensors["normalized_key"]
                    .reshape(1, len(query_positions), key_heads, head_dimensions)
                    .transpose(1, 2),
                    cosine,
                    sine,
                )
                differences["query_checkpoint_rotary"] = compare_tensors(
                    tensors["rotated_query"],
                    query_rotary.transpose(1, 2).reshape_as(query),
                )
                differences["key_checkpoint_rotary"] = compare_tensors(
                    tensors["rotated_key"], key_rotary.transpose(1, 2).reshape_as(key)
                )
                current_key = observed["keys"][index].to(reference.device)
                current_value = observed["values"][index].to(reference.device)
                if not full_prefix:
                    current_key, current_value = current_key[-1:], current_value[-1:]
                else:
                    pass
                assert (
                    current_key.shape
                    == current_value.shape
                    == (len(query_positions), key_heads, head_dimensions)
                )
                assert current_key.dtype == current_value.dtype == torch.bfloat16
                assert (
                    native_keys[index].shape
                    == native_values[index].shape
                    == (
                        int(before["positions"][-1]) + (frame == 0),
                        key_heads,
                        head_dimensions,
                    )
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
                if frame:
                    native_keys[index] = torch.cat((native_keys[index], current_key))
                    native_values[index] = torch.cat(
                        (native_values[index], current_value)
                    )
                else:
                    pass
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
                    .reshape(1, len(query_positions), query_heads, head_dimensions)
                    .transpose(1, 2)
                    .double()
                )
                scores = (queries @ keys.transpose(-1, -2)) * head_dimensions**-0.5
                future = (
                    torch.arange(keys.shape[-2], device=reference.device)[None, :]
                    > query_positions.to(reference.device)[:, None]
                )
                attention = (
                    torch.softmax(
                        scores.masked_fill(future[None, None], float("-inf")), dim=-1
                    )
                    @ values
                )
                differences["attention_float64_rounded"] = compare_tensors(
                    tensors["attention_values"],
                    attention.transpose(1, 2)
                    .reshape(len(query_positions), -1)
                    .to(reference.dtype),
                )
                differences["attention_projection_replay"] = compare_tensors(
                    tensors["attention_output"],
                    layer.self_attn.o_proj(tensors["attention_values"]),
                )
                logits = tensors["router_logits"]
                selected = tensors["expert_indices"].long()
                assert (selected.sort(dim=-1).values.diff(dim=-1) != 0).all()
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
                        final_hidden[-1:],
                    )
                    captured_logits = (
                        torch.load(
                            directory / "0000-prefill-logits.pt", weights_only=True
                        )
                        if frame == 0
                        else observed["logits"]
                    )
                    head_logits = reference.codec_head(
                        observed["hidden_states"].to(reference.device).unsqueeze(0)
                    ).float()
                    assert head_logits.isfinite().all()
                    if frame == 0:
                        suppressed = torch.zeros_like(captured_logits, dtype=torch.bool)
                        suppressed[sampling.suppress_tokens] = True
                        assert torch.equal(captured_logits.isneginf(), suppressed)
                        assert not suppressed[configuration.codec_eos_token_id]
                        captured_logits = captured_logits[~suppressed]
                        head_logits = head_logits[:, (~suppressed).to(reference.device)]
                    else:
                        pass
                    differences["head_reprojection"] = compare_tensors(
                        captured_logits.to(reference.device).unsqueeze(0), head_logits
                    )
                else:
                    pass
                for name, difference in differences.items():
                    exact_counts[name] = exact_counts.get(name, 0) + difference.exact
                    maximum_errors[name] = max(
                        maximum_errors.get(name, 0), difference.maximum_absolute_error
                    )
                layer_rows += len(query_positions)
                layer_steps += 1
                previous_snapshot = snapshot
                with (destination / "layers.jsonl").open("a") as output:
                    output.write(
                        json.dumps(
                            {
                                "frame": frame,
                                "layer": index,
                                "query_positions": query_positions.tolist(),
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
            "decode_frames": sum(
                int(filename.name.split("-")[0]) > 0 for filename in files
            ),
            "prefill_frames": int(arguments.include_prefill),
            "layer_rows": layer_rows,
            "layer_steps": layer_steps,
            "exact_counts_unit": "whole operator output tensors, not individual token rows",
            "full_prefill_prefix": arguments.full_prefill_prefix,
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
