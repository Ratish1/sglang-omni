"""Compare actual Thinker operands without propagating earlier layer differences."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    rotate_half,
)

from scripts.debug_qwen3_omni_feedback_audit import TensorDifference, compare_tensors


@dataclass(kw_only=True)
class ThinkerOperatorCapture:
    layers: dict[str, torch.Tensor]
    operators: dict[str, torch.Tensor]
    positions: torch.Tensor
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    batch_offset: int
    prefix_token_count: int
    extend_token_count: int


@dataclass(kw_only=True)
class ThinkerLayerComparison:
    layer_index: int
    comparisons: dict[str, TensorDifference]
    expert_membership_exact: bool
    cache_store_exact: bool
    residual_assembly_exact: bool


def fp32_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    values = hidden_states.float()
    inverse_rms = torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    return (values * inverse_rms * weight.float()).to(weight.dtype)


@torch.inference_mode()
def audit_thinker_operators(
    reference: Qwen3OmniMoeThinkerForConditionalGeneration,
    captured: ThinkerOperatorCapture,
    output_directory: Path,
) -> None:
    device = reference.device
    layers = {name: tensor.to(device) for name, tensor in captured.layers.items()}
    operators = {name: tensor.to(device) for name, tensor in captured.operators.items()}
    live = slice(
        captured.batch_offset, captured.batch_offset + captured.extend_token_count
    )
    initial = layers["layer_00_hidden"]
    positions = captured.positions.to(device).T[:, None]
    cosine, sine = reference.model.rotary_emb(initial[None], positions)
    cosine = cosine[0, :, None].float()
    sine = sine[0, :, None].float()
    unrounded_input = initial.float()
    comparisons = []
    for layer_index in range(len(layers) - 1):
        layer = reference.model.layers[layer_index]
        attention = layer.self_attn
        prefix = f"operator_{layer_index:02d}"
        hidden = layers[f"layer_{layer_index:02d}_hidden"]
        assert torch.equal(hidden[live], unrounded_input.to(hidden.dtype)[live])
        normalized = operators[f"{prefix}_qkv_input"]
        projected = operators[f"{prefix}_attention_projection_output"]
        attention_input = operators[f"{prefix}_attention_projection_input"]
        router_input = operators[f"{prefix}_router_input"]
        logits = operators[f"{prefix}_router_output"]
        expert_indices = operators[f"{prefix}_expert_indices"].long()
        expert_weights = operators[f"{prefix}_expert_weights"]
        assert int(expert_indices.min()) >= 0
        assert int(expert_indices.max()) < layer.mlp.experts.num_experts
        packed_weights = torch.cat(
            (attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight)
        )
        query_features = attention.q_proj.weight.shape[0]
        key_features = attention.k_proj.weight.shape[0]
        packed = operators[f"{prefix}_qkv_output"]
        query, key, value = packed.split(
            (query_features, key_features, key_features), dim=-1
        )
        query = query.reshape(len(query), -1, attention.head_dim)
        key = key.reshape(len(key), -1, attention.head_dim)
        normalized_query = fp32_norm(
            query, attention.q_norm.weight, attention.q_norm.variance_epsilon
        )
        normalized_key = fp32_norm(
            key, attention.k_norm.weight, attention.k_norm.variance_epsilon
        )
        rotated_query = (
            normalized_query.float() * cosine
            + rotate_half(normalized_query).float() * sine
        ).to(query.dtype)
        rotated_key = (
            normalized_key.float() * cosine + rotate_half(normalized_key).float() * sine
        ).to(key.dtype)
        actual_query = operators[f"{prefix}_query_rotated"].reshape_as(query)
        actual_key = operators[f"{prefix}_key_rotated"].reshape_as(key)
        native_keys = captured.keys[layer_index].to(device)
        native_values = captured.values[layer_index].to(device)
        assert torch.equal(actual_key[live], native_keys[captured.prefix_token_count :])
        assert torch.equal(
            operators[f"{prefix}_value"][live].reshape_as(
                native_values[captured.prefix_token_count :]
            ),
            native_values[captured.prefix_token_count :],
        )
        assert torch.equal(
            value[live].reshape_as(native_values[captured.prefix_token_count :]),
            native_values[captured.prefix_token_count :],
        )
        key_states = native_keys.transpose(0, 1).repeat_interleave(
            attention.num_key_value_groups, dim=0
        )
        value_states = native_values.transpose(0, 1).repeat_interleave(
            attention.num_key_value_groups, dim=0
        )
        scores = (
            actual_query[live].transpose(0, 1).float()
            @ key_states.float().transpose(1, 2)
        ) * attention.scaling
        query_positions = captured.prefix_token_count + torch.arange(
            captured.extend_token_count, device=device
        )
        allowed = (
            torch.arange(len(native_keys), device=device)[None]
            <= query_positions[:, None]
        )
        scores.masked_fill_(~allowed[None], float("-inf"))
        mathematical_attention = (
            (scores.softmax(-1) @ value_states.float())
            .transpose(0, 1)
            .reshape(captured.extend_token_count, query_features)
            .to(hidden.dtype)
        )
        probabilities = logits.float().softmax(-1)
        selected_weights = probabilities.gather(1, expert_indices)
        selected_weights /= selected_weights.sum(-1, keepdim=True)
        selected_indices = probabilities.topk(expert_indices.shape[1], dim=-1).indices
        membership = torch.equal(
            selected_indices[live].sort(-1).values,
            expert_indices[live].sort(-1).values,
        )
        reference_moe = layer.mlp.experts(router_input, expert_indices, expert_weights)
        unrounded_attention_residual = hidden.float() + projected.float()
        next_unrounded = (
            unrounded_attention_residual.to(hidden.dtype).float()
            + operators[f"{prefix}_moe_output"].float()
        )
        assert torch.equal(
            next_unrounded.to(hidden.dtype)[live],
            layers[f"layer_{layer_index + 1:02d}_hidden"][live],
        )
        differences = {
            "input_norm_fp32_formula": compare_tensors(
                normalized[live],
                fp32_norm(
                    unrounded_input,
                    layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon,
                )[live],
            ),
            "packed_qkv_checkpoint": compare_tensors(
                packed[live],
                torch.nn.functional.linear(normalized, packed_weights)[live],
            ),
            "query_norm_rope_fp32_formula": compare_tensors(
                actual_query[live], rotated_query[live]
            ),
            "key_norm_rope_fp32_formula": compare_tensors(
                actual_key[live], rotated_key[live]
            ),
            "attention_fp32_formula": compare_tensors(
                attention_input[live], mathematical_attention
            ),
            "attention_projection_checkpoint": compare_tensors(
                projected[live],
                torch.nn.functional.linear(attention_input, attention.o_proj.weight)[
                    live
                ],
            ),
            "ffn_norm_fp32_formula": compare_tensors(
                router_input[live],
                fp32_norm(
                    unrounded_attention_residual,
                    layer.post_attention_layernorm.weight,
                    layer.post_attention_layernorm.variance_epsilon,
                )[live],
            ),
            "router_checkpoint": compare_tensors(
                logits[live],
                torch.nn.functional.linear(router_input, layer.mlp.gate.weight)[live],
            ),
            "router_weights_fp32_formula": compare_tensors(
                expert_weights[live], selected_weights[live]
            ),
            "experts_checkpoint_native_routing": compare_tensors(
                operators[f"{prefix}_moe_output"][live], reference_moe[live]
            ),
        }
        comparisons.append(
            ThinkerLayerComparison(
                layer_index=layer_index,
                comparisons=differences,
                expert_membership_exact=membership,
                cache_store_exact=True,
                residual_assembly_exact=True,
            )
        )
        unrounded_input = next_unrounded
    report = {
        "scope": "Actual native operands, padded projection geometry, cached prefix KV and selected expert routing. FP32 formulas and checkpoint kernels can have different valid rounding. No speech-quality repair is inferred from a numerical distance.",
        "layers": [asdict(comparison) for comparison in comparisons],
    }
    (output_directory / "operator-comparisons.json").write_text(
        json.dumps(report, indent=2)
    )
