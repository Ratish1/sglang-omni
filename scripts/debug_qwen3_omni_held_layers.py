# SPDX-License-Identifier: Apache-2.0
"""Audit the first Talker decode step with saved native inputs and KV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import linear
from transformers import Qwen3OmniMoeConfig
from transformers.cache_utils import DynamicCache
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    apply_rotary_pos_emb,
)

from scripts.debug_qwen3_omni_held_talker import filtered_distribution


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, int | float]:
    assert actual.numel() == expected.numel()
    actual = actual.reshape(-1).float()
    expected = expected.reshape(-1).float()
    assert actual.isfinite().all() and expected.isfinite().all()
    delta = actual - expected
    return {
        "different_elements": int(torch.count_nonzero(delta)),
        "maximum_absolute_error": float(delta.abs().max()),
        "relative_l2_error": float(delta.norm() / expected.norm().clamp_min(1e-12)),
    }


def normalization_differences(
    observed: torch.Tensor,
    contribution: torch.Tensor,
    previous_residual: torch.Tensor | None,
    recorded_residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> dict[str, dict[str, int | float]]:
    full_residual = contribution.double()
    if previous_residual is not None:
        full_residual += previous_residual.double()
    else:
        pass
    assert torch.equal(full_residual.to(contribution.dtype), recorded_residual)
    reports = {}
    for residual_name, residual in (
        ("full_residual", full_residual),
        ("rounded_residual", recorded_residual.double()),
    ):
        normalized = residual * torch.rsqrt(
            residual.square().mean(-1, keepdim=True) + epsilon
        )
        outputs = {
            "weight_before_cast": (normalized * weight.double()).to(observed.dtype),
            "weight_after_cast": (normalized.to(observed.dtype) * weight).to(
                observed.dtype
            ),
        }
        for cast_name, expected in outputs.items():
            reports[f"{residual_name}_{cast_name}"] = difference(observed, expected)
    return reports


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    target_device = torch.device("cuda", 0)
    names = (
        ("h200_prefix", arguments.fixture, "first-decode-prefix-cache-000.pt"),
        ("h200_operands", arguments.fixture, "backbone-decode-001.pt"),
        ("h200_sampling", arguments.fixture, "decode-sampling-001.pt"),
        ("h100_sampling", arguments.native, "decode-sampling-001.pt"),
        ("h100_held", arguments.native, "held-first-decode-000.pt"),
        ("h100_layers", arguments.native, "first-decode-layers.pt"),
    )
    tensors = {
        name: torch.load(
            directory / filename, map_location=target_device, weights_only=True
        )
        for name, directory, filename in names
    }
    held = tensors["h100_held"]
    h100_sampling = tensors["h100_sampling"]
    h200_sampling = tensors["h200_sampling"]
    snapshots = tensors["h100_layers"]
    prefix = tensors["h200_prefix"]
    position = h200_sampling["position"].reshape(1, 1, 1)
    assert int(position) == held["position"] == prefix["position"]
    embedding = held["embedding"]
    operands = tensors["h200_operands"]
    assert torch.equal(
        embedding,
        operands["feedback_row"].reshape(-1) + operands["text_row"].reshape(-1),
    )
    assert torch.equal(embedding, snapshots[0]["input_contribution"])
    assert all(
        int(layer["positions"].reshape(-1)[0]) == int(position) for layer in snapshots
    )
    assert all(
        torch.equal(h200_sampling[key], h100_sampling[key])
        for key in (
            "position",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "suppress_mask",
            "repetition_mask",
            "repetition_penalty",
        )
    )

    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    reference.eval()
    assert len(snapshots) == len(reference.model.layers) == len(prefix["key_rows"])
    settings = config.text_config
    query_heads = settings.num_attention_heads
    key_heads = settings.num_key_value_heads
    head_dimension = settings.head_dim
    position_axes = operands["rotary_positions"]
    linear_position = position.reshape(1, 1, 1)
    rotary_positions = (
        linear_position.expand(3, 1, 1)
        if position_axes is None
        else position_axes.reshape(3, 1, 1)
    )
    cache = DynamicCache(config=settings)
    for layer_index, (key_rows, value_rows) in enumerate(
        zip(prefix["key_rows"], prefix["value_rows"], strict=True)
    ):
        cache.update(
            key_rows.transpose(0, 1).unsqueeze(0),
            value_rows.transpose(0, 1).unsqueeze(0),
            layer_index,
        )
    reference_output = reference.model(
        inputs_embeds=embedding.view(1, 1, -1),
        position_ids=torch.cat((linear_position, rotary_positions)),
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
    )
    assert cache.get_seq_length() == int(position) + 1
    cosines, sines = reference.model.rotary_emb(
        embedding.view(1, 1, -1), rotary_positions
    )
    reference_logits = (
        reference.codec_head(reference_output.last_hidden_state).float().reshape(-1)
    )
    distributions = {}
    selections = {}
    for source, logits in (
        ("h200_native", h200_sampling["raw_logits"]),
        ("h100_native", h100_sampling["raw_logits"]),
        ("h100_checkpoint", reference_logits),
    ):
        distributions[source], selections[source] = filtered_distribution(
            logits, h200_sampling
        )
    assert selections["h100_native"] == int(h100_sampling["sampled_code"])
    assert selections["h200_native"] == int(h200_sampling["sampled_code"])
    native_selected = selections["h200_native"]
    replayed_native_head = reference.codec_head(
        h100_sampling["predictor_query"]
    ).float()

    layers = []
    for layer_index, (layer, snapshot) in enumerate(
        zip(reference.model.layers, snapshots, strict=True)
    ):
        rows = {name: value.reshape(1, -1) for name, value in snapshot.items()}
        if layer_index:
            assert torch.equal(
                snapshot["input_contribution"], snapshots[layer_index - 1]["mlp_output"]
            )
            assert torch.equal(
                snapshot["previous_residual"],
                snapshots[layer_index - 1]["mlp_residual"],
            )
        else:
            assert "previous_residual" not in snapshot
        attention = layer.self_attn
        query, key, value = rows["qkv"].split(
            (
                query_heads * head_dimension,
                key_heads * head_dimension,
                key_heads * head_dimension,
            ),
            dim=-1,
        )
        checkpoint_query_rope, checkpoint_key_rope = apply_rotary_pos_emb(
            rows["query_norm"].view(1, query_heads, 1, head_dimension),
            rows["key_norm"].view(1, key_heads, 1, head_dimension),
            cosines,
            sines,
        )
        saved_keys = torch.cat(
            (
                prefix["key_rows"][layer_index],
                rows["key_rope"].view(1, key_heads, head_dimension),
            )
        )
        saved_values = torch.cat(
            (
                prefix["value_rows"][layer_index],
                rows["value"].view(1, key_heads, head_dimension),
            )
        )
        keys = (
            saved_keys.transpose(0, 1)
            .double()
            .repeat_interleave(query_heads // key_heads, dim=0)
        )
        values = (
            saved_values.transpose(0, 1)
            .double()
            .repeat_interleave(query_heads // key_heads, dim=0)
        )
        queries = rows["query_rope"].view(query_heads, 1, head_dimension).double()
        scores = (queries @ keys.transpose(-1, -2)) * head_dimension**-0.5
        attention_context = ((scores.softmax(-1) @ values).reshape(1, -1)).to(
            reference.dtype
        )
        assert torch.equal(rows["value"], value)
        source = rows["mlp_normalized"]
        expert_router_logits, expert_weights, expert_indices = layer.mlp.gate(source)
        probabilities = (
            expert_router_logits.float().softmax(-1).gather(1, expert_indices)
        )
        if settings.norm_topk_prob:
            probabilities /= probabilities.sum(-1, keepdim=True)
        else:
            pass
        shared = layer.mlp.shared_expert(source)
        assert torch.equal(
            rows["shared_before_gate"] * rows["shared_gate"].sigmoid(),
            rows["shared_output"],
        )
        assert torch.equal(
            rows["routed_output"] + rows["shared_output"], rows["mlp_output"]
        )
        comparisons = {
            "chained_layer_input": difference(
                rows["attention_residual"], reference_output.hidden_states[layer_index]
            ),
            "qkv_projection": difference(
                rows["qkv"],
                linear(
                    rows["attention_normalized"],
                    torch.cat(
                        (
                            attention.q_proj.weight,
                            attention.k_proj.weight,
                            attention.v_proj.weight,
                        )
                    ),
                ),
            ),
            "query_norm": difference(
                rows["query_norm"], attention.q_norm(query.reshape(-1, head_dimension))
            ),
            "key_norm": difference(
                rows["key_norm"], attention.k_norm(key.reshape(-1, head_dimension))
            ),
            "query_rotary": difference(rows["query_rope"], checkpoint_query_rope),
            "key_rotary": difference(rows["key_rope"], checkpoint_key_rope),
            "attention_context_float64": difference(
                rows["attention_context"], attention_context
            ),
            "attention_output": difference(
                rows["attention_output"], attention.o_proj(rows["attention_context"])
            ),
            "router_logits": difference(rows["router_logits"], expert_router_logits),
            "shared_gate": difference(
                rows["shared_gate"], layer.mlp.shared_expert_gate(source)
            ),
            "shared_expert": difference(rows["shared_before_gate"], shared),
            "routed_checkpoint_bfloat16_weights": difference(
                rows["routed_output"],
                layer.mlp.experts(source, expert_indices, expert_weights),
            ),
            "routed_checkpoint_float32_weights": difference(
                rows["routed_output"],
                layer.mlp.experts(source, expert_indices, probabilities),
            ),
        }
        layers.append(
            {
                "layer": layer_index,
                "operators_on_same_input": comparisons,
                "attention_normalization": normalization_differences(
                    rows["attention_normalized"],
                    rows["input_contribution"],
                    rows.get("previous_residual"),
                    rows["attention_residual"],
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                ),
                "mlp_normalization": normalization_differences(
                    rows["mlp_normalized"],
                    rows["attention_output"],
                    rows["attention_residual"],
                    rows["mlp_residual"],
                    layer.post_attention_layernorm.weight,
                    layer.post_attention_layernorm.variance_epsilon,
                ),
            }
        )
    report = {
        "device": torch.cuda.get_device_name(target_device),
        "position": int(position),
        "selected_codes": selections,
        "probability_of_h200_code": {
            source: float(distribution[0, native_selected])
            for source, distribution in distributions.items()
        },
        "h100_native_vs_h200_native_logits": difference(
            h100_sampling["raw_logits"], h200_sampling["raw_logits"]
        ),
        "h100_native_vs_checkpoint_logits": difference(
            h100_sampling["raw_logits"], reference_logits
        ),
        "h100_native_head_reprojection": difference(
            h100_sampling["raw_logits"], replayed_native_head
        ),
        "final_normalization": normalization_differences(
            h100_sampling["predictor_query"],
            snapshots[-1]["mlp_output"],
            snapshots[-1]["mlp_residual"],
            (
                snapshots[-1]["mlp_output"].double()
                + snapshots[-1]["mlp_residual"].double()
            ).to(reference.dtype),
            reference.model.norm.weight,
            reference.model.norm.variance_epsilon,
        ),
        "layers": layers,
    }
    with arguments.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
else:
    pass
