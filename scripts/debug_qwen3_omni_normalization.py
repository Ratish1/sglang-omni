# SPDX-License-Identifier: Apache-2.0
"""Isolate native normalization arithmetic in checkpoint Talker generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from torch.utils.hooks import RemovableHandle
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeTalkerTextSparseMoeBlock,
    Qwen3OmniMoeTextRMSNorm,
    Qwen3OmniMoeThinkerTextAttention,
    Qwen3OmniMoeThinkerTextRMSNorm,
)


@dataclass(kw_only=True)
class NativeNormalization:
    reference: Qwen3OmniMoeTalkerForConditionalGeneration
    enabled: bool = True
    attention_residual: torch.Tensor | None = None
    attention_output: torch.Tensor | None = None
    mlp_residual: torch.Tensor | None = None
    mlp_output: torch.Tensor | None = None
    boundary_calls: int = 0
    query_key_calls: int = 0
    handles: list[RemovableHandle] = field(default_factory=list)

    def install(self) -> None:
        for layer in self.reference.model.layers:
            self.handles.extend(
                [
                    layer.input_layernorm.register_forward_hook(self.input_norm),
                    layer.self_attn.register_forward_hook(self.capture_attention),
                    layer.post_attention_layernorm.register_forward_hook(
                        self.post_attention_norm
                    ),
                    layer.mlp.register_forward_hook(self.capture_mlp),
                    layer.self_attn.q_norm.register_forward_hook(self.query_key_norm),
                    layer.self_attn.k_norm.register_forward_hook(self.query_key_norm),
                ]
            )
        self.handles.append(
            self.reference.model.norm.register_forward_hook(self.final_norm)
        )

    def normalize_sum(
        self,
        contribution: torch.Tensor,
        residual: torch.Tensor,
        rounded_sum: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        normalized = contribution.reshape(-1, weight.numel()).clone()
        updated_residual = residual.reshape_as(normalized).clone()
        fused_add_rmsnorm(normalized, updated_residual, weight, epsilon)
        assert torch.equal(updated_residual.view_as(rounded_sum), rounded_sum)
        self.boundary_calls += 1
        return normalized.view_as(rounded_sum)

    def input_norm(
        self,
        module: Qwen3OmniMoeThinkerTextRMSNorm,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return output
        else:
            pass
        self.attention_residual = inputs[0]
        if module is self.reference.model.layers[0].input_layernorm:
            self.boundary_calls += 1
            return rmsnorm(
                inputs[0].reshape(-1, module.weight.numel()),
                module.weight,
                module.variance_epsilon,
            ).view_as(output)
        else:
            assert self.mlp_output is not None and self.mlp_residual is not None
            return self.normalize_sum(
                self.mlp_output,
                self.mlp_residual,
                inputs[0],
                module.weight,
                module.variance_epsilon,
            )

    def capture_attention(
        self,
        module: Qwen3OmniMoeThinkerTextAttention,
        inputs: tuple[()],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        self.attention_output = output[0]

    def post_attention_norm(
        self,
        module: Qwen3OmniMoeThinkerTextRMSNorm,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return output
        else:
            pass
        assert self.attention_output is not None
        assert self.attention_residual is not None
        self.mlp_residual = inputs[0]
        return self.normalize_sum(
            self.attention_output,
            self.attention_residual,
            inputs[0],
            module.weight,
            module.variance_epsilon,
        ).view_as(output)

    def capture_mlp(
        self,
        module: Qwen3OmniMoeTalkerTextSparseMoeBlock,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        self.mlp_output = output

    def query_key_norm(
        self,
        module: Qwen3OmniMoeThinkerTextRMSNorm,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return output
        else:
            pass
        self.query_key_calls += 1
        return rmsnorm(
            inputs[0].reshape(-1, module.weight.numel()),
            module.weight,
            module.variance_epsilon,
        ).view_as(output)

    def final_norm(
        self,
        module: Qwen3OmniMoeTextRMSNorm,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled:
            return output
        else:
            pass
        assert self.mlp_output is not None and self.mlp_residual is not None
        return self.normalize_sum(
            self.mlp_output,
            self.mlp_residual,
            inputs[0],
            module.weight,
            module.variance_epsilon,
        ).view_as(output)
