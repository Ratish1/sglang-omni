"""Isolate packed projections and paired normalization in Talker recurrence."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import torch
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from torch.utils.hooks import RemovableHandle
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextRMSNorm,
)

PROJECTION_NAMES = ("query", "key", "value")


@dataclass(kw_only=True)
class PackedAttentionProjection:
    projected: torch.Tensor
    normalized_query: torch.Tensor | None = None
    normalized_key: torch.Tensor | None = None


@dataclass(kw_only=True)
class NativeAttentionInputs:
    reference: Qwen3OmniMoeTalkerForConditionalGeneration
    enabled: bool = False
    mode: Literal["merged", "paired", "merged-paired"] = "merged"
    projection_calls: int = 0
    normalization_calls: int = 0
    weights: list[torch.Tensor] = field(default_factory=list)
    projections: dict[int, PackedAttentionProjection] = field(default_factory=dict)
    handles: list[RemovableHandle] = field(default_factory=list)

    def install(self) -> None:
        for layer_index, layer in enumerate(self.reference.model.layers):
            attention = layer.self_attn
            linear_projections = (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
            )
            assert all(projection.bias is None for projection in linear_projections)
            assert all(
                projection.weight.dtype == torch.bfloat16
                for projection in linear_projections
            )
            assert (
                attention.q_norm.weight.dtype
                == attention.k_norm.weight.dtype
                == torch.bfloat16
            )
            assert (
                attention.q_norm.variance_epsilon == attention.k_norm.variance_epsilon
            )
            self.weights.append(
                torch.cat([projection.weight for projection in linear_projections])
                .detach()
                .contiguous()
            )
            for name, projection in zip(
                PROJECTION_NAMES, linear_projections, strict=True
            ):
                self.handles.append(
                    projection.register_forward_hook(
                        partial(self.project, layer_index=layer_index, name=name)
                    )
                )
            for name, normalization in (
                ("query", attention.q_norm),
                ("key", attention.k_norm),
            ):
                self.handles.append(
                    normalization.register_forward_hook(
                        partial(self.normalize, layer_index=layer_index, name=name)
                    )
                )

    def project(
        self,
        module: torch.nn.Linear,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
        *,
        layer_index: int,
        name: Literal["query", "key", "value"],
    ) -> torch.Tensor:
        if not self.enabled:
            return output
        else:
            pass
        assert layer_index == (self.projection_calls // len(PROJECTION_NAMES)) % len(
            self.weights
        )
        assert name == PROJECTION_NAMES[self.projection_calls % len(PROJECTION_NAMES)]
        attention = self.reference.model.layers[layer_index].self_attn
        query_dimensions = attention.q_proj.out_features
        key_dimensions = attention.k_proj.out_features
        value_dimensions = attention.v_proj.out_features
        if name == "query":
            assert module is attention.q_proj
            hidden_states = inputs[0].reshape(-1, inputs[0].shape[-1])
            if self.mode in ("merged", "merged-paired"):
                projected = torch.nn.functional.linear(
                    hidden_states, self.weights[layer_index]
                )
            else:
                projected = torch.cat(
                    (
                        output.reshape(-1, query_dimensions),
                        torch.nn.functional.linear(
                            hidden_states, attention.k_proj.weight
                        ),
                        torch.nn.functional.linear(
                            hidden_states, attention.v_proj.weight
                        ),
                    ),
                    dim=-1,
                )
            assert projected.is_contiguous() and projected.dtype == torch.bfloat16
            state = PackedAttentionProjection(projected=projected)
            if self.mode in ("paired", "merged-paired"):
                normalized = projected.clone()
                head_dimensions = attention.head_dim
                query = normalized.as_strided(
                    (
                        len(normalized),
                        query_dimensions // head_dimensions,
                        head_dimensions,
                    ),
                    (normalized.stride(0), head_dimensions, 1),
                )
                key = normalized.as_strided(
                    (
                        len(normalized),
                        key_dimensions // head_dimensions,
                        head_dimensions,
                    ),
                    (normalized.stride(0), head_dimensions, 1),
                    storage_offset=query_dimensions,
                )
                fused_inplace_qknorm(
                    q=query,
                    k=key,
                    q_weight=attention.q_norm.weight,
                    k_weight=attention.k_norm.weight,
                    eps=attention.q_norm.variance_epsilon,
                    head_dim=head_dimensions,
                )
                state.normalized_query = query
                state.normalized_key = key
            else:
                pass
            assert layer_index not in self.projections
            self.projections[layer_index] = state
        else:
            state = self.projections[layer_index]
        query, key, value = state.projected.split(
            (query_dimensions, key_dimensions, value_dimensions), dim=-1
        )
        selected = {"query": query, "key": key, "value": value}[name].view_as(output)
        if self.mode == "paired":
            assert torch.equal(selected, output)
        else:
            pass
        if name == "value":
            self.projections.pop(layer_index)
        else:
            pass
        self.projection_calls += 1
        return selected

    def normalize(
        self,
        module: Qwen3OmniMoeThinkerTextRMSNorm,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
        *,
        layer_index: int,
        name: Literal["query", "key"],
    ) -> torch.Tensor:
        if not self.enabled or self.mode == "merged":
            return output
        else:
            pass
        attention = self.reference.model.layers[layer_index].self_attn
        expected = attention.q_norm if name == "query" else attention.k_norm
        assert module is expected and inputs[0].dtype == torch.bfloat16
        assert layer_index == (self.normalization_calls // 2) % len(self.weights)
        assert name == PROJECTION_NAMES[self.normalization_calls % 2]
        state = self.projections[layer_index]
        normalized = state.normalized_query if name == "query" else state.normalized_key
        assert normalized is not None
        self.normalization_calls += 1
        return normalized.reshape_as(output)
