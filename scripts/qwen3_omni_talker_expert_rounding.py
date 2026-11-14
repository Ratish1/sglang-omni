"""Compare FlashInfer-style and checkpoint expert rounding on held operands."""

from __future__ import annotations

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerTextExperts,
)


def expert_rounding_stages(
    experts: Qwen3OmniMoeTalkerTextExperts,
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    assert hidden_states.dtype == experts.gate_up_proj.dtype == torch.bfloat16
    assert expert_weights.dtype == torch.float32
    route_shape = (*expert_indices.shape, hidden_states.shape[-1])
    fused_activation_routes = torch.zeros(
        route_shape, device=hidden_states.device, dtype=hidden_states.dtype
    )
    rounded_activation_routes = torch.zeros_like(fused_activation_routes)
    rounded_weighted_routes = torch.zeros_like(fused_activation_routes)
    checkpoint_output = torch.zeros_like(hidden_states)
    fused_activation_checkpoint_output = torch.zeros_like(hidden_states)
    for expert_index in expert_indices.unique(sorted=True).tolist():
        ranks, token_indices = torch.where(expert_indices.T == expert_index)
        gate, up = torch.nn.functional.linear(
            hidden_states[token_indices], experts.gate_up_proj[expert_index]
        ).chunk(2, dim=-1)
        fused_activation = (torch.nn.functional.silu(gate.float()) * up.float()).to(
            hidden_states.dtype
        )
        rounded_activation = torch.nn.functional.silu(gate) * up
        fused_down = torch.nn.functional.linear(
            fused_activation, experts.down_proj[expert_index]
        )
        rounded_down = torch.nn.functional.linear(
            rounded_activation, experts.down_proj[expert_index]
        )
        weighted = (rounded_down * expert_weights[token_indices, ranks, None]).to(
            hidden_states.dtype
        )
        fused_activation_routes[token_indices, ranks] = fused_down
        rounded_activation_routes[token_indices, ranks] = rounded_down
        rounded_weighted_routes[token_indices, ranks] = weighted
        checkpoint_output.index_add_(0, token_indices, weighted)
        fused_activation_checkpoint_output.index_add_(
            0,
            token_indices,
            (fused_down * expert_weights[token_indices, ranks, None]).to(
                hidden_states.dtype
            ),
        )
    stage_outputs = {}
    for name, routes in (
        ("flashinfer_fma", fused_activation_routes),
        ("rounded_activation_fma", rounded_activation_routes),
    ):
        accumulated = torch.zeros_like(hidden_states, dtype=torch.float32)
        for rank in range(expert_indices.shape[1]):
            accumulated = torch.addcmul(
                accumulated,
                routes[:, rank].float(),
                expert_weights[:, rank, None],
            )
        stage_outputs[name] = accumulated.to(hidden_states.dtype)
    separate_accumulated = torch.zeros_like(hidden_states, dtype=torch.float32)
    weighted_accumulated = torch.zeros_like(hidden_states, dtype=torch.float32)
    for rank in range(expert_indices.shape[1]):
        separate_accumulated = separate_accumulated + (
            fused_activation_routes[:, rank].float() * expert_weights[:, rank, None]
        )
        weighted_accumulated = (
            weighted_accumulated + rounded_weighted_routes[:, rank].float()
        )
    stage_outputs["flashinfer_separate_multiply_add"] = separate_accumulated.to(
        hidden_states.dtype
    )
    stage_outputs["rounded_weighted_store"] = weighted_accumulated.to(
        hidden_states.dtype
    )
    stage_outputs["checkpoint_expert_order_bfloat16_sum"] = checkpoint_output
    stage_outputs["fused_activation_checkpoint_sum"] = (
        fused_activation_checkpoint_output
    )
    return stage_outputs
