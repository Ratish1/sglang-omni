"""Reconstruct and verify captured native Talker expert parameters."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from pydantic import BaseModel
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


class WeightFingerprint(BaseModel):
    shape: list[int]
    dtype: str
    sha256: str


class LayerRuntime(BaseModel):
    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int
    weights: dict[str, WeightFingerprint]


class TacticChoice(BaseModel):
    operation: str
    tensor_shapes: list[list[int]]
    tactic: int
    capturing: bool
    tuning: bool


class NativeRuntime(BaseModel):
    tune_max_num_tokens: int
    use_fused_finalize: bool
    layers: list[LayerRuntime]
    tactic_choices: list[TacticChoice]


class ObservedExpertProfiles(BaseModel):
    source_capture: Path
    prefill: list[tuple[int, int]]
    decode: list[tuple[int, int]]
    verified_frames: int


class ObservedExpertProfileMap(BaseModel):
    profiles: dict[str, ObservedExpertProfiles]
    scope: str


def native_expert_contributions(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    packed_weights: torch.Tensor,
    down_weights: torch.Tensor,
    runtime: NativeRuntime,
    profiles: tuple[int, int],
) -> list[torch.Tensor]:
    assert not runtime.use_fused_finalize
    contributions = []
    for slot in range(indices.shape[1]):
        replay_inputs = hidden_states.contiguous().clone()
        contributions.append(
            cutlass_fused_moe(
                input=replay_inputs,
                token_selected_experts=indices[:, slot : slot + 1]
                .to(torch.int32)
                .contiguous(),
                token_final_scales=torch.ones(
                    (len(hidden_states), 1),
                    dtype=torch.float32,
                    device=hidden_states.device,
                ),
                fc1_expert_weights=packed_weights,
                fc2_expert_weights=down_weights,
                output_dtype=torch.bfloat16,
                quant_scales=None,
                activation_type=ActivationType.Swiglu,
                tune_max_num_tokens=runtime.tune_max_num_tokens,
                use_fused_finalize=runtime.use_fused_finalize,
                profile_ids=list(profiles),
            )[0]
        )
        assert torch.equal(replay_inputs, hidden_states)
    return contributions


def combine_expert_contributions(
    contributions: list[torch.Tensor],
    indices: torch.Tensor,
    weights: torch.Tensor,
    semantics: Literal[
        "native", "rounded-contributions", "bf16-reduction", "checkpoint-reduction"
    ],
) -> torch.Tensor:
    assert len(contributions) == indices.shape[1] == weights.shape[1]
    assert torch.equal(weights, weights.bfloat16().float())
    selected = torch.stack(contributions, dim=1)
    if semantics == "checkpoint-reduction":
        order = indices.argsort(dim=1)
        selected = selected.gather(1, order[:, :, None].expand_as(selected))
        weights = weights.gather(1, order)
    else:
        pass
    dtype = (
        torch.float32
        if semantics in ("native", "rounded-contributions")
        else torch.bfloat16
    )
    combined = torch.zeros_like(contributions[0], dtype=dtype)
    for slot in range(len(contributions)):
        weighted = selected[:, slot].float() * weights[:, slot : slot + 1]
        if semantics == "native":
            combined += weighted
        else:
            combined += weighted.bfloat16().to(dtype)
    return combined.bfloat16()


def load_native_expert_weights(
    reference: Qwen3OmniMoeTalkerForConditionalGeneration,
    runtime: NativeRuntime,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    native_weights = []
    for layer, layer_runtime in zip(
        reference.model.layers, runtime.layers, strict=True
    ):
        assert layer_runtime.tp_size == layer_runtime.ep_size == 1
        assert layer_runtime.tp_rank == layer_runtime.ep_rank == 0
        gate, up = layer.mlp.experts.gate_up_proj.chunk(2, dim=1)
        packed = torch.cat((up, gate), dim=1).contiguous()
        down = layer.mlp.experts.down_proj.contiguous()
        for name, tensor in (("w13_weight", packed), ("w2_weight", down)):
            fingerprint = layer_runtime.weights[name]
            assert list(tensor.shape) == fingerprint.shape
            assert str(tensor.dtype) == fingerprint.dtype == "torch.bfloat16"
            assert (
                hashlib.sha256(
                    tensor.detach().cpu().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                == fingerprint.sha256
            )
        native_weights.append((packed, down))
    return native_weights
