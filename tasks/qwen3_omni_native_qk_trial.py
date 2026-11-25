# SPDX-License-Identifier: Apache-2.0
"""Isolate checkpoint Q/K normalization on selected native decode rows."""

import json
from dataclasses import asdict, dataclass
from os import environ
from pathlib import Path
from typing import Literal

import torch
from sglang.kernels.ops.layernorm.rmsnorm_hf import rmsnorm_hf
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch import nn

from sglang_omni.models.qwen3_omni.components.thinker_model import (
    Qwen3OmniMoeThinkerTextAttention,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.layers import RMSNorm


@dataclass(kw_only=True)
class QueryRestore:
    request_id: str
    frame_index: int
    layer_index: int


def checkpoint_normalize(
    hidden_states: torch.Tensor,
    normalization: RMSNorm,
    head_channels: int,
    fused: bool,
) -> torch.Tensor:
    if fused:
        return rmsnorm_hf(
            hidden_states.reshape(-1, head_channels).contiguous(),
            normalization.weight,
            normalization.variance_epsilon,
        ).reshape_as(hidden_states)
    else:
        heads = hidden_states.reshape(hidden_states.shape[0], -1, head_channels)
        expanded = heads.float()
        variance = expanded.pow(2).mean(-1, keepdim=True)
        normalized = expanded * torch.rsqrt(variance + normalization.variance_epsilon)
        return (normalization.weight * normalized.to(heads.dtype)).reshape_as(
            hidden_states
        )


class NativeQkLayerTrial:
    def __init__(
        self,
        attention: Qwen3OmniMoeThinkerTextAttention,
        selected_rows: torch.Tensor,
        fused: bool,
        restore_rows: torch.Tensor | None,
    ) -> None:
        assert not attention.use_fused_qk_norm_rope
        self.attention: Qwen3OmniMoeThinkerTextAttention = attention
        self.selected_rows: torch.Tensor = selected_rows
        self.fused: bool = fused
        self.restore_rows: torch.Tensor | None = restore_rows
        self.is_decode: bool = False
        self.projection: torch.Tensor | None = None
        attention.register_forward_pre_hook(self.begin, with_kwargs=True)
        attention.register_forward_hook(self.finish)
        attention.qkv_proj.register_forward_hook(self.save_projection)
        attention.rotary_emb.register_forward_pre_hook(self.before_rotary)

    def begin(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        named_inputs: dict[str, torch.Tensor | ForwardBatch],
    ) -> None:
        forward_batch = named_inputs["forward_batch"]
        assert isinstance(forward_batch, ForwardBatch)
        self.is_decode = forward_batch.forward_mode.is_decode()
        self.projection = None

    def finish(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        # note (ratish): rotary modules are shared across layers and predictor calls.
        self.is_decode = False
        self.projection = None

    def save_projection(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        if self.is_decode:
            # note (ratish): native fused Q/K normalization can overwrite this projection.
            self.projection = output[0].clone()
        else:
            pass

    def before_rotary(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not self.is_decode:
            return None
        else:
            assert self.projection is not None
        positions, query, key = inputs
        attention = self.attention
        raw_query, raw_key = self.projection.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
        )[:2]
        assert query.shape == raw_query.shape and key.shape == raw_key.shape
        assert query.dtype == key.dtype == torch.bfloat16
        rows = query.shape[0]
        assert rows <= self.selected_rows.numel()
        mask = self.selected_rows[:rows, None]
        normalized_query = checkpoint_normalize(
            raw_query, attention.q_norm, attention.head_dim, self.fused
        )
        if self.restore_rows is not None:
            assert self.fused
            normalized_query = torch.where(
                self.restore_rows[:rows, None],
                checkpoint_normalize(
                    raw_query, attention.q_norm, attention.head_dim, False
                ),
                normalized_query,
            )
        else:
            pass
        return (
            positions,
            torch.where(mask, normalized_query, query),
            torch.where(
                mask,
                checkpoint_normalize(
                    raw_key, attention.k_norm, attention.head_dim, self.fused
                ),
                key,
            ),
        )


class NativeQkTrial:
    def __init__(
        self,
        mode: Literal["control", "checkpoint", "fused_checkpoint"],
        request_ids: set[str],
        max_batch_rows: int,
        device: torch.device,
        attentions: list[Qwen3OmniMoeThinkerTextAttention],
    ) -> None:
        assert mode in ("control", "checkpoint", "fused_checkpoint") and request_ids
        self.mode: Literal["control", "checkpoint", "fused_checkpoint"] = mode
        self.request_ids: set[str] = request_ids
        self.selected_rows: torch.Tensor = torch.zeros(
            max_batch_rows, dtype=torch.bool, device=device
        )
        self.restore: QueryRestore | None = None
        self.restore_rows: torch.Tensor | None = None
        restore_specification = environ.get("OMNI_NATIVE_QK_RESTORE_QUERY", "")
        if restore_specification:
            sample_id, frame_index, layer_index = restore_specification.split(",")
            self.restore = QueryRestore(
                request_id=f"feedback-trace-native-{sample_id}",
                frame_index=int(frame_index),
                layer_index=int(layer_index),
            )
            assert mode == "fused_checkpoint"
            assert self.restore.request_id in request_ids
            assert self.restore.frame_index > 0
            assert 0 <= self.restore.layer_index < len(attentions)
            self.restore_rows = torch.zeros_like(self.selected_rows)
        else:
            pass
        self.layers: list[NativeQkLayerTrial] = [
            NativeQkLayerTrial(
                attention,
                self.selected_rows,
                mode == "fused_checkpoint",
                (
                    self.restore_rows
                    if self.restore is not None
                    and self.restore.layer_index == layer_index
                    else None
                ),
            )
            for layer_index, attention in enumerate(attentions)
        ]

    def select(self, requests: list[SchedulerRequest]) -> None:
        assert len(requests) <= self.selected_rows.numel()
        self.selected_rows.zero_()
        if self.mode != "control":
            self.selected_rows[: len(requests)].copy_(
                torch.tensor(
                    [request.request_id in self.request_ids for request in requests],
                    dtype=torch.bool,
                    device=self.selected_rows.device,
                )
            )
        else:
            pass
        if self.restore is not None:
            assert self.restore_rows is not None
            self.restore_rows.zero_()
            for row_index, request in enumerate(requests):
                if request.request_id == self.restore.request_id:
                    assert isinstance(request.data, SGLangARRequestData)
                    assert request.data.req is not None
                    if len(request.data.req.output_ids) == self.restore.frame_index:
                        self.restore_rows[row_index] = True
                        destination = Path(environ["OMNI_NATIVE_TRAJECTORY_DIRECTORY"])
                        with (destination / "query-restore-selection.json").open(
                            "x"
                        ) as receipt:
                            json.dump(
                                {**asdict(self.restore), "request_row": row_index},
                                receipt,
                                indent=2,
                            )
                    else:
                        pass
                else:
                    pass
        else:
            pass
