"""Isolate native routed experts in otherwise checkpoint Talker recurrence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from torch.utils.hooks import RemovableHandle
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeTalkerTextExperts,
    Qwen3OmniMoeTalkerTextTopKRouter,
)

from scripts.debug_qwen3_omni_moe_batches import NativeRuntime


@dataclass(kw_only=True)
class NativeTalkerExperts:
    reference: Qwen3OmniMoeTalkerForConditionalGeneration
    capture_directory: Path
    route_precision: Literal["checkpoint", "float32"]
    enabled: bool = False
    calls: int = 0
    router_calls: int = 0
    runtime: NativeRuntime = field(init=False)
    weights: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    handles: list[RemovableHandle] = field(default_factory=list)

    def install(self) -> None:
        self.runtime = NativeRuntime.model_validate_json(
            (self.capture_directory / "moe-runtime.json").read_bytes()
        )
        for layer_index, (layer, recorded) in enumerate(
            zip(self.reference.model.layers, self.runtime.layers, strict=True)
        ):
            assert recorded.tp_size == recorded.ep_size == 1
            assert recorded.tp_rank == recorded.ep_rank == 0
            gate, up = layer.mlp.experts.gate_up_proj.chunk(2, dim=1)
            packed = torch.cat((up, gate), dim=1).contiguous()
            down = layer.mlp.experts.down_proj.contiguous()
            for name, tensor in (("w13_weight", packed), ("w2_weight", down)):
                fingerprint = recorded.weights[name]
                assert list(tensor.shape) == fingerprint.shape
                assert str(tensor.dtype) == fingerprint.dtype == "torch.bfloat16"
                assert (
                    hashlib.sha256(
                        tensor.detach().cpu().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    == fingerprint.sha256
                )
            self.weights.append((packed, down))
            self.handles.extend(
                [
                    layer.mlp.gate.register_forward_hook(self.routing),
                    layer.mlp.experts.register_forward_hook(
                        partial(self.experts, layer_index=layer_index)
                    ),
                ]
            )

    def routing(
        self,
        module: Qwen3OmniMoeTalkerTextTopKRouter,
        inputs: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enabled and self.route_precision == "float32":
            logits, _, indices = output
            probabilities = logits.float().softmax(dim=-1)
            weights = probabilities.gather(1, indices)
            if module.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            else:
                pass
            self.router_calls += 1
            return logits, weights, indices
        else:
            return output

    def experts(
        self,
        module: Qwen3OmniMoeTalkerTextExperts,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        if not self.enabled:
            return output
        else:
            pass
        hidden_states, indices, weights = inputs
        assert hidden_states.dtype == torch.bfloat16
        assert indices.shape == weights.shape
        assert weights.dtype == (
            torch.float32 if self.route_precision == "float32" else torch.bfloat16
        )
        packed, down = self.weights[layer_index]
        replay_inputs = hidden_states.contiguous().clone()
        fused_output = cutlass_fused_moe(
            input=replay_inputs,
            token_selected_experts=indices.to(torch.int32).contiguous(),
            token_final_scales=weights.float().contiguous(),
            fc1_expert_weights=packed,
            fc2_expert_weights=down,
            output_dtype=torch.bfloat16,
            quant_scales=None,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=self.runtime.tune_max_num_tokens,
            use_fused_finalize=self.runtime.use_fused_finalize,
            profile_ids=[-1, -1],
        )[0]
        assert torch.equal(replay_inputs, hidden_states)
        assert fused_output.shape == output.shape
        self.calls += 1
        return fused_output
