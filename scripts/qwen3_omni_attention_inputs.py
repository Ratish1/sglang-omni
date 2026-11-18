"""Isolate packed projections and paired normalization in Talker recurrence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal

import torch
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf
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
    is_checkpoint_query_key_enabled: bool = False
    native_directory: Path | None = None
    verify_native_inputs: bool = False
    head_calls: int = 0
    output_calls: int = 0
    cached_frame_index: int = -1
    frame_snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)
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
            self.handles.append(
                attention.o_proj.register_forward_hook(
                    partial(self.project_output, layer_index=layer_index)
                )
            )
        self.handles.append(
            self.reference.codec_head.register_forward_hook(self.project_head)
        )

    def snapshot(self, layer_index: int) -> dict[str, torch.Tensor]:
        assert self.native_directory is not None
        if self.cached_frame_index != self.head_calls:
            self.frame_snapshots = torch.load(
                self.native_directory / f"{self.head_calls:04d}-operators.pt",
                weights_only=True,
            )
            assert len(self.frame_snapshots) == len(self.weights)
            self.cached_frame_index = self.head_calls
        else:
            pass
        return self.frame_snapshots[layer_index]

    def project_batch(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        layer_index: int,
        input_name: Literal["attention_normalized", "attention_values"],
    ) -> torch.Tensor:
        snapshot = self.snapshot(layer_index)
        operands = snapshot[f"batch_{input_name}"].to(hidden_states.device)
        if self.head_calls == 0:
            start, stop = snapshot["prefix_token_rows"].tolist()
        else:
            start = int(snapshot["batch_selected_row"])
            stop = start + 1
        selected = hidden_states.reshape(-1, hidden_states.shape[-1])
        assert operands[start:stop].shape == selected.shape
        assert operands.dtype == selected.dtype == weight.dtype == torch.bfloat16
        layout = snapshot[f"layout_{input_name}"].tolist()
        assert layout == [*operands.shape, *operands.stride(), operands.data_ptr() % 16]
        if self.verify_native_inputs:
            assert torch.equal(operands[start:stop], selected), (
                self.native_directory,
                self.head_calls,
                layer_index,
                input_name,
            )
        else:
            pass
        operands[start:stop] = selected
        return torch.nn.functional.linear(operands, weight)[start:stop]

    def project_output(
        self,
        module: torch.nn.Linear,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
        *,
        layer_index: int,
    ) -> torch.Tensor:
        if not self.enabled or self.native_directory is None:
            return output
        else:
            pass
        assert module is self.reference.model.layers[layer_index].self_attn.o_proj
        projected = self.project_batch(
            inputs[0], module.weight, layer_index, "attention_values"
        )
        self.output_calls += 1
        return projected.view_as(output)

    def project_head(
        self,
        module: torch.nn.Linear,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enabled or self.native_directory is None:
            return output
        else:
            pass
        assert module is self.reference.codec_head
        snapshot = self.snapshot(len(self.weights) - 1)
        head_inputs = snapshot["batch_codec_head_input"].to(inputs[0].device)
        metadata = json.loads(
            (self.native_directory / f"{self.head_calls:04d}-batch.json").read_text()
        )
        row = metadata["active_row"]
        assert isinstance(row, int) and 0 <= row < len(head_inputs)
        selected = inputs[0].reshape(-1, inputs[0].shape[-1])[-1]
        assert (
            head_inputs.dtype == selected.dtype == module.weight.dtype == torch.bfloat16
        )
        if self.verify_native_inputs:
            assert torch.equal(head_inputs[row], selected), (
                self.native_directory,
                self.head_calls,
                "codec_head",
            )
        else:
            pass
        head_inputs[row] = selected
        projected = torch.nn.functional.linear(head_inputs, module.weight)[row]
        assert output.is_contiguous() and output.shape[:-1] == inputs[0].shape[:-1]
        replacement = output.clone()
        replacement.reshape(-1, output.shape[-1])[-1] = projected
        self.head_calls += 1
        return replacement

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
                if self.native_directory is not None:
                    projected = self.project_batch(
                        hidden_states,
                        self.weights[layer_index],
                        layer_index,
                        "attention_normalized",
                    )
                else:
                    projected = torch.nn.functional.linear(
                        hidden_states, self.weights[layer_index]
                    )
            elif self.native_directory is not None:
                projected = torch.cat(
                    [
                        self.project_batch(
                            hidden_states,
                            projection.weight,
                            layer_index,
                            "attention_normalized",
                        )
                        for projection in (
                            attention.q_proj,
                            attention.k_proj,
                            attention.v_proj,
                        )
                    ],
                    dim=-1,
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
                if self.is_checkpoint_query_key_enabled:
                    query.copy_(
                        rmsnorm_hf(
                            query.reshape(-1, head_dimensions).contiguous(),
                            attention.q_norm.weight,
                            attention.q_norm.variance_epsilon,
                        ).view_as(query)
                    )
                    key.copy_(
                        rmsnorm_hf(
                            key.reshape(-1, head_dimensions).contiguous(),
                            attention.k_norm.weight,
                            attention.k_norm.variance_epsilon,
                        ).view_as(key)
                    )
                else:
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
        if self.mode == "paired" and self.native_directory is None:
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
