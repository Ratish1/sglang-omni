"""Apply measured native query/key arithmetic inside a reference attention layer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm

from scripts.debug_qwen3_omni_feedback_audit import TensorDifference, compare_tensors
from scripts.debug_qwen3_omni_thinker_rotary import contracted_bf16_rotation
from scripts.qwen3_omni_thinker_operator_audit import fp32_norm


@dataclass(kw_only=True)
class NativeRotaryControl:
    positions: torch.Tensor
    table: torch.Tensor
    sections: list[int]
    head_features: int
    epsilon: float
    query_norm_weight: torch.Tensor | None = None
    key_norm_weight: torch.Tensor | None = None
    query_projection: torch.Tensor | None = None
    key_projection: torch.Tensor | None = None
    calls: int = 0
    normalization_comparisons: dict[str, TensorDifference] = field(default_factory=dict)

    def record_query_projection(
        self,
        projection: torch.nn.Linear,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        assert inputs[0].shape[-1] == projection.in_features
        assert output.shape[-1] == projection.out_features
        assert self.query_projection is None
        self.query_projection = output.detach().clone()

    def record_key_projection(
        self,
        projection: torch.nn.Linear,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        assert inputs[0].shape[-1] == projection.in_features
        assert output.shape[-1] == projection.out_features
        assert self.key_projection is None
        self.key_projection = output.detach().clone()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.calls == 0
        assert query.dtype == key.dtype == cosine.dtype == sine.dtype == torch.bfloat16
        assert query.shape[0] == key.shape[0] == 1
        if self.query_norm_weight is not None:
            assert self.key_norm_weight is not None
            assert self.query_projection is not None and self.key_projection is not None
            query_rows = self.query_projection.reshape(
                query.shape[2], query.shape[1], self.head_features
            ).clone()
            key_rows = self.key_projection.reshape(
                key.shape[2], key.shape[1], self.head_features
            ).clone()
            expected_query = fp32_norm(query_rows, self.query_norm_weight, self.epsilon)
            expected_key = fp32_norm(key_rows, self.key_norm_weight, self.epsilon)
            fused_inplace_qknorm(
                query_rows,
                key_rows,
                self.query_norm_weight,
                self.key_norm_weight,
                self.epsilon,
                head_dim=self.head_features,
            )
            self.normalization_comparisons = {
                "query": compare_tensors(query_rows, expected_query),
                "key": compare_tensors(key_rows, expected_key),
            }
            query = query_rows.unsqueeze(0).transpose(1, 2)
            key = key_rows.unsqueeze(0).transpose(1, 2)
        else:
            assert self.key_norm_weight is None
        cosine = cosine.unsqueeze(unsqueeze_dim)
        sine = sine.unsqueeze(unsqueeze_dim)
        expected_query, expected_key = (
            contracted_bf16_rotation(query, cosine, sine),
            contracted_bf16_rotation(key, cosine, sine),
        )
        query_rows = query.transpose(1, 2).reshape(query.shape[2], -1).clone()
        key_rows = key.transpose(1, 2).reshape(key.shape[2], -1).clone()
        triton_mrope_fused(
            query_rows,
            key_rows,
            self.table,
            self.positions,
            self.sections,
            self.head_features,
            self.head_features,
            True,
            False,
            True,
            None,
        )
        query_output = query_rows.reshape(
            1, query.shape[2], query.shape[1], query.shape[3]
        ).transpose(1, 2)
        key_output = key_rows.reshape(
            1, key.shape[2], key.shape[1], key.shape[3]
        ).transpose(1, 2)
        assert torch.equal(query_output, expected_query)
        assert torch.equal(key_output, expected_key)
        self.calls += 1
        return query_output, key_output
