"""Hold residual ownership while applying native normalization in a reference layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRMSNorm,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
)


@dataclass(kw_only=True)
class NativeResidualControl:
    residual: torch.Tensor | None = None
    contribution: torch.Tensor | None = None
    projection: torch.Tensor | None = None
    input_norm_calls: int = 0
    post_attention_norm_calls: int = 0

    def input_norm(
        self,
        normalization: Qwen3OmniMoeThinkerTextRMSNorm,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        hidden = inputs[0]
        assert hidden.shape == output.shape and hidden.shape[0] == 1
        rows = hidden.reshape(-1, hidden.shape[-1])
        if self.residual is None:
            assert self.input_norm_calls == 0
            self.residual = rows.clone()
            normalized = rmsnorm(
                rows, normalization.weight, normalization.variance_epsilon
            )
        else:
            assert self.contribution is not None
            assert torch.equal(
                rows,
                (self.contribution.float() + self.residual.float()).to(rows.dtype),
            )
            normalized = self.contribution.clone()
            fused_add_rmsnorm(
                normalized,
                self.residual,
                normalization.weight,
                normalization.variance_epsilon,
            )
            assert torch.equal(self.residual, rows)
        self.projection = None
        self.input_norm_calls += 1
        return normalized.reshape_as(hidden)

    def record_projection(
        self,
        projection: torch.nn.Linear,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        assert inputs[0].shape[-1] == projection.in_features
        assert output.shape[-1] == projection.out_features
        assert self.projection is None
        self.projection = output.reshape(-1, output.shape[-1]).clone()

    def post_attention_norm(
        self,
        normalization: Qwen3OmniMoeThinkerTextRMSNorm,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        hidden = inputs[0]
        assert hidden.shape == output.shape
        assert self.projection is not None and self.residual is not None
        rows = hidden.reshape(-1, hidden.shape[-1])
        assert torch.equal(
            rows, (self.projection.float() + self.residual.float()).to(rows.dtype)
        )
        normalized = self.projection.clone()
        fused_add_rmsnorm(
            normalized,
            self.residual,
            normalization.weight,
            normalization.variance_epsilon,
        )
        assert torch.equal(self.residual, rows)
        self.post_attention_norm_calls += 1
        return normalized.reshape_as(hidden)

    def record_contribution(
        self,
        mixture: Qwen3OmniMoeThinkerTextSparseMoeBlock,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        assert inputs[0].shape[-1] == mixture.gate.hidden_dim
        assert inputs[0].shape == output.shape
        self.contribution = output.reshape(-1, output.shape[-1]).clone()

    def check_layer(self, hidden: torch.Tensor) -> None:
        assert self.contribution is not None and self.residual is not None
        assert self.input_norm_calls == self.post_attention_norm_calls
        assert torch.equal(
            hidden.reshape_as(self.residual),
            (self.contribution.float() + self.residual.float()).to(hidden.dtype),
        )
