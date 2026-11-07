# SPDX-License-Identifier: Apache-2.0
"""Audit captured Talker operators with identical inputs and checkpoint weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
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


def norm_values(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    output_dtype: torch.dtype,
    *,
    round_before_weight: bool,
) -> torch.Tensor:
    normalized = hidden_states.float() * torch.rsqrt(
        hidden_states.float().square().mean(-1, keepdim=True) + epsilon
    )
    if round_before_weight:
        return (normalized.to(output_dtype) * weight.to(output_dtype)).to(output_dtype)
    else:
        return (normalized * weight.float()).to(output_dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-directory", action="append", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    assert not torch.backends.cuda.matmul.allow_tf32
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    query_heads = config.text_config.num_attention_heads
    key_heads = config.text_config.num_key_value_heads
    head_dimensions = config.text_config.head_dim
    reports = []
    with torch.inference_mode():
        for directory in arguments.trace_directory:
            fingerprints = json.loads((directory / "parameter-sha256.json").read_text())
            weight_mismatches = []
            for parameter_name, expected_hash in fingerprints.items():
                layer_index = int(parameter_name.split(".")[1])
                layer = reference.model.layers[layer_index]
                if parameter_name.endswith("self_attn.qkv_proj.weight"):
                    parameter = torch.cat(
                        [
                            layer.self_attn.q_proj.weight,
                            layer.self_attn.k_proj.weight,
                            layer.self_attn.v_proj.weight,
                        ]
                    )
                elif parameter_name.endswith("mlp.shared_expert.gate_up_proj.weight"):
                    parameter = torch.cat(
                        [
                            layer.mlp.shared_expert.gate_proj.weight,
                            layer.mlp.shared_expert.up_proj.weight,
                        ]
                    )
                else:
                    parameter = reference.model.get_parameter(parameter_name)
                actual_hash = hashlib.sha256(
                    parameter.detach()
                    .cpu()
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                ).hexdigest()
                if actual_hash != expected_hash:
                    weight_mismatches.append(parameter_name)
                else:
                    pass
            before_prefill = torch.load(
                next(directory.glob("*-before_prefill.pt")),
                map_location="cpu",
                weights_only=True,
            )
            previous_keys: dict[int, torch.Tensor] = {}
            previous_values: dict[int, torch.Tensor] = {}
            frames = []
            for trace_file in sorted(directory.glob("*-post_*.pt")):
                captured = torch.load(trace_file, map_location="cpu", weights_only=True)
                is_prefill = "post_prefill" in trace_file.name
                before_file = trace_file.with_name(
                    trace_file.name.replace("post_", "before_")
                )
                before = torch.load(before_file, map_location="cpu", weights_only=True)
                active_rows = len(before["positions"])
                if is_prefill:
                    matching_counts = [
                        int(name.rsplit("_", 1)[1])
                        for name, tensor in captured.items()
                        if name.startswith("boundary_0_attention_input_contribution_")
                        and tensor.shape[0] >= active_rows
                        and torch.equal(
                            tensor[:active_rows], before_prefill["prefill_embeds"]
                        )
                    ]
                    assert len(matching_counts) == 1, matching_counts
                    token_count = matching_counts[0]
                else:
                    token_count = 1
                positions = before["positions"].to(reference.device)
                cosine, sine = reference.model.rotary_emb(
                    torch.empty(
                        1,
                        active_rows,
                        config.text_config.hidden_size,
                        dtype=torch.bfloat16,
                        device=reference.device,
                    ),
                    positions.view(1, 1, -1).expand(3, 1, -1),
                )
                layers = []
                for layer_index, layer in enumerate(reference.model.layers):
                    boundary = f"boundary_{layer_index}_"
                    attention = f"attention_{layer_index}_"
                    differences: dict[str, TensorDifference] = {}
                    for stage, weight in (
                        ("attention", layer.input_layernorm.weight),
                        ("mlp", layer.post_attention_layernorm.weight),
                    ):
                        contribution = captured[
                            f"{boundary}{stage}_input_contribution_{token_count}"
                        ][:active_rows].to(reference.device)
                        if stage == "attention":
                            previous_name = (
                                f"{boundary}attention_previous_residual_{token_count}"
                            )
                            previous_residual = (
                                captured[previous_name][:active_rows].to(
                                    reference.device
                                )
                                if previous_name in captured
                                else None
                            )
                        else:
                            previous_residual = captured[
                                f"{boundary}attention_residual_{token_count}"
                            ][:active_rows].to(reference.device)
                        unrounded = (
                            contribution.float()
                            if previous_residual is None
                            else contribution.float() + previous_residual.float()
                        )
                        rounded = unrounded.to(contribution.dtype)
                        actual_residual = captured[
                            f"{boundary}{stage}_residual_{token_count}"
                        ][:active_rows].to(reference.device)
                        actual_norm = captured[
                            f"{boundary}{stage}_normalized_{token_count}"
                        ][:active_rows].to(reference.device)
                        differences[f"{stage}_residual"] = compare_tensors(
                            actual_residual, rounded
                        )
                        for variant, operands, round_before_weight in (
                            ("unrounded_single_round", unrounded, False),
                            ("rounded_single_round", rounded, False),
                            ("checkpoint_rounding", rounded, True),
                        ):
                            differences[f"{stage}_norm_{variant}"] = compare_tensors(
                                actual_norm,
                                norm_values(
                                    operands,
                                    weight,
                                    config.text_config.rms_norm_eps,
                                    actual_norm.dtype,
                                    round_before_weight=round_before_weight,
                                ),
                            )
                    projection_input = captured[
                        f"{attention}projection_input_{token_count}"
                    ][:active_rows].to(reference.device)
                    projection_output = captured[
                        f"{attention}projection_output_{token_count}"
                    ][:active_rows].to(reference.device)
                    projection_weight = torch.cat(
                        [
                            layer.self_attn.q_proj.weight,
                            layer.self_attn.k_proj.weight,
                            layer.self_attn.v_proj.weight,
                        ]
                    )
                    differences["qkv_bfloat16"] = compare_tensors(
                        projection_output,
                        torch.nn.functional.linear(projection_input, projection_weight),
                    )
                    differences["qkv_float32_rounded"] = compare_tensors(
                        projection_output,
                        torch.nn.functional.linear(
                            projection_input.float(), projection_weight.float()
                        ).to(projection_output.dtype),
                    )
                    raw_query, raw_key, raw_value = projection_output.split(
                        [
                            query_heads * head_dimensions,
                            key_heads * head_dimensions,
                            key_heads * head_dimensions,
                        ],
                        dim=-1,
                    )
                    normalized_query = (
                        captured[f"{attention}normalized_query_{token_count}"][
                            :active_rows
                        ]
                        .to(reference.device)
                        .reshape(active_rows, query_heads, head_dimensions)
                    )
                    normalized_key = (
                        captured[f"{attention}normalized_key_{token_count}"][
                            :active_rows
                        ]
                        .to(reference.device)
                        .reshape(active_rows, key_heads, head_dimensions)
                    )
                    for name, raw, actual, weight in (
                        (
                            "query",
                            raw_query,
                            normalized_query,
                            layer.self_attn.q_norm.weight,
                        ),
                        ("key", raw_key, normalized_key, layer.self_attn.k_norm.weight),
                    ):
                        raw = raw.reshape(active_rows, -1, head_dimensions)
                        for variant, round_before_weight in (
                            ("single_round", False),
                            ("checkpoint_rounding", True),
                        ):
                            differences[f"{name}_norm_{variant}"] = compare_tensors(
                                actual,
                                norm_values(
                                    raw,
                                    weight,
                                    config.text_config.rms_norm_eps,
                                    actual.dtype,
                                    round_before_weight=round_before_weight,
                                ),
                            )
                    expected_query, expected_key = apply_rotary_pos_emb(
                        normalized_query.transpose(0, 1)[None],
                        normalized_key.transpose(0, 1)[None],
                        cosine,
                        sine,
                    )
                    query = (
                        captured[f"{attention}rotated_query_{token_count}"][
                            :active_rows
                        ]
                        .to(reference.device)
                        .reshape(active_rows, query_heads, head_dimensions)
                    )
                    current_key = (
                        captured[f"{attention}cached_key_{token_count}"][:active_rows]
                        .to(reference.device)
                        .reshape(active_rows, key_heads, head_dimensions)
                    )
                    current_value = (
                        captured[f"{attention}cached_value_{token_count}"][:active_rows]
                        .to(reference.device)
                        .reshape(active_rows, key_heads, head_dimensions)
                    )
                    differences["rope_query_checkpoint"] = compare_tensors(
                        query, expected_query[0].transpose(0, 1)
                    )
                    differences["rope_key_checkpoint"] = compare_tensors(
                        current_key, expected_key[0].transpose(0, 1)
                    )
                    differences["stored_value"] = compare_tensors(
                        current_value, raw_value.reshape_as(current_value)
                    )
                    if is_prefill:
                        key_prefix = current_key
                        value_prefix = current_value
                    else:
                        prefix_length = int(captured[f"{attention}prefix_lengths_1"][0])
                        assert prefix_length == int(positions[0]) + 1
                        key_prefix = captured[f"{attention}prefix_key_1"][
                            :prefix_length
                        ].to(reference.device)
                        value_prefix = captured[f"{attention}prefix_value_1"][
                            :prefix_length
                        ].to(reference.device)
                        assert key_prefix.shape[0] == prefix_length
                        differences["retained_key_prefix"] = compare_tensors(
                            key_prefix[:-1], previous_keys[layer_index]
                        )
                        differences["retained_value_prefix"] = compare_tensors(
                            value_prefix[:-1], previous_values[layer_index]
                        )
                        differences["current_key_in_prefix"] = compare_tensors(
                            key_prefix[-1:], current_key
                        )
                        differences["current_value_in_prefix"] = compare_tensors(
                            value_prefix[-1:], current_value
                        )
                    previous_keys[layer_index] = key_prefix.clone()
                    previous_values[layer_index] = value_prefix.clone()
                    query_rows = query.transpose(0, 1)[None]
                    key_rows = key_prefix.transpose(0, 1)[None]
                    value_rows = value_prefix.transpose(0, 1)[None]
                    actual_attention = captured[
                        f"{attention}attention_output_{token_count}"
                    ][:active_rows].to(reference.device)
                    for dtype in (torch.bfloat16, torch.float32):
                        expected_attention = (
                            torch.nn.functional.scaled_dot_product_attention(
                                query_rows.to(dtype),
                                key_rows.to(dtype),
                                value_rows.to(dtype),
                                is_causal=is_prefill,
                                enable_gqa=query_heads != key_heads,
                            )
                            .transpose(1, 2)
                            .reshape_as(actual_attention)
                            .to(actual_attention.dtype)
                        )
                        differences[f"attention_{str(dtype)}"] = compare_tensors(
                            actual_attention, expected_attention
                        )
                    actual_projection = captured[
                        f"{attention}output_projection_{token_count}"
                    ][:active_rows].to(reference.device)
                    differences["output_projection"] = compare_tensors(
                        actual_projection, layer.self_attn.o_proj(actual_attention)
                    )
                    differences["output_projection_float32_rounded"] = compare_tensors(
                        actual_projection,
                        torch.nn.functional.linear(
                            actual_attention.float(),
                            layer.self_attn.o_proj.weight.float(),
                        ).to(actual_projection.dtype),
                    )
                    mlp_input = captured[f"{boundary}mlp_normalized_{token_count}"][
                        :active_rows
                    ].to(reference.device)
                    mlp_output = captured[f"{boundary}mlp_output_{token_count}"][
                        :active_rows
                    ].to(reference.device)
                    differences["whole_mlp_checkpoint_bfloat16"] = compare_tensors(
                        mlp_output, layer.mlp(mlp_input[None])[0]
                    )
                    layers.append(
                        {
                            "layer_index": layer_index,
                            "differences": {
                                name: asdict(difference)
                                for name, difference in differences.items()
                            },
                        }
                    )
                frames.append(
                    {
                        "trace_file": trace_file.name,
                        "active_rows": active_rows,
                        "captured_token_count": token_count,
                        "layers": layers,
                    }
                )
                print(
                    json.dumps(
                        {
                            "trace_directory": str(directory),
                            "trace_file": trace_file.name,
                            "layers": len(layers),
                        }
                    ),
                    flush=True,
                )
            reports.append(
                {
                    "trace_directory": str(directory),
                    "parameter_fingerprints_checked": len(fingerprints),
                    "parameter_fingerprint_mismatches": weight_mismatches,
                    "frames": frames,
                }
            )
            (arguments.output_directory / "results.json").write_text(
                json.dumps(reports, indent=2)
            )


if __name__ == "__main__":
    main()
