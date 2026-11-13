"""Reconstruct and verify captured native Talker expert parameters."""

from __future__ import annotations

import hashlib

import torch
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
