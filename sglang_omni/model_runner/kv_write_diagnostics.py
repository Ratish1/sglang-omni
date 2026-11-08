"""Capture KV writes and attention operands inside model CUDA graphs."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum

import torch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from torch import nn

from sglang_omni.vendor.sglang.layers import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

logger = logging.getLogger(__name__)


class HeadPrecisionMode(StrEnum):
    ONLY = "only"
    EXCEPT = "except"


@dataclass(kw_only=True)
class HeadPrecisionControl:
    request_id: str
    frame_index: int
    group_index: int
    mode: HeadPrecisionMode
    mask: torch.Tensor
    activations: int = 0

    def reset(self) -> None:
        self.mask.fill_(self.mode is HeadPrecisionMode.EXCEPT)

    def select(self, request_ids: list[str], frame_indices: list[int]) -> None:
        self.reset()
        for row, (request_id, frame_index) in enumerate(
            zip(request_ids, frame_indices, strict=True)
        ):
            if request_id == self.request_id and frame_index == self.frame_index:
                self.mask[row, self.group_index - 1] = (
                    self.mode is HeadPrecisionMode.ONLY
                )
                self.activations += 1
                logger.info(
                    f"Head precision control request={request_id} frame={frame_index} "
                    f"group={self.group_index} mode={self.mode} activations={self.activations}"
                )
            else:
                pass


class HeadPrecisionHook:
    def __call__(
        self,
        linear: ReplicatedLinear,
        arguments: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert linear.bias is None
        unrounded = torch.mm(
            arguments[0].reshape(-1, linear.weight.shape[1]),
            linear.weight.T,
            out_dtype=torch.float32,
        ).reshape_as(output[0])
        selected = linear.diagnostic_head_precision_mask[: output[0].shape[0]]
        return (
            torch.where(selected[:, None, None], unrounded, output[0].float()),
            output[1],
        )


class ProjectionSnapshotHook:
    def __init__(
        self,
        query_index: torch.Tensor,
        input_snapshot: torch.Tensor | None,
        output_snapshot: torch.Tensor,
    ) -> None:
        self.query_index = query_index
        self.input_snapshot = input_snapshot
        self.output_snapshot = output_snapshot

    def __call__(
        self,
        linear: QKVParallelLinear | RowParallelLinear,
        arguments: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        if self.input_snapshot is not None:
            self.input_snapshot.copy_(
                arguments[0].index_select(0, self.query_index).view(-1)
            )
        else:
            pass
        self.output_snapshot.copy_(output[0].index_select(0, self.query_index).view(-1))


class CodecLogitHook:
    def __call__(
        self,
        linear: ReplicatedLinear,
        arguments: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        logits = output[0]
        linear.diagnostic_last_logits[: logits.shape[0]].copy_(logits)


class KVWriteHook:
    def __call__(
        self,
        attention: RadixAttention,
        arguments: tuple[torch.Tensor, torch.Tensor, torch.Tensor, ForwardBatch],
        output: torch.Tensor,
    ) -> None:
        query, keys, values, forward_batch = arguments
        locations = forward_batch.out_cache_loc.long()
        pool = get_token_to_kv_pool()
        stored_keys, stored_values = pool.get_kv_buffer(attention.layer_id)
        valid = locations > 0
        wrong_keys = (
            stored_keys.index_select(0, locations)
            != keys.reshape(-1, attention.tp_k_head_num, attention.head_dim)
        ).any(dim=(1, 2))
        wrong_values = (
            stored_values.index_select(0, locations)
            != values.reshape(-1, attention.tp_v_head_num, attention.v_head_dim)
        ).any(dim=(1, 2))
        attention.diagnostic_kv_write_counts.add_(
            torch.stack(
                [(wrong_keys & valid).sum(), (wrong_values & valid).sum(), valid.sum()]
            )
        )
        last_query_index = attention.diagnostic_query_index
        attention.diagnostic_query.copy_(
            query.index_select(0, last_query_index).view(
                attention.tp_q_head_num, attention.head_dim
            )
        )
        attention.diagnostic_attention_output.copy_(
            output.index_select(0, last_query_index).view(
                attention.tp_q_head_num, attention.v_head_dim
            )
        )
        attention.diagnostic_position.copy_(
            forward_batch.positions.index_select(0, last_query_index)[0]
        )


def install_kv_write_observers(
    layers: nn.ModuleList, *, capture_moe: bool = False
) -> None:
    if os.environ.get("SGLANG_OMNI_DIAGNOSTIC_KV_DIRECTORY") is None:
        return
    else:
        pass
    for layer in layers:
        attention = layer.self_attn.attn
        assert isinstance(attention, RadixAttention)
        device = layer.self_attn.q_norm.weight.device
        for name, shape, dtype in (
            ("diagnostic_kv_write_counts", (3,), torch.int64),
            (
                "diagnostic_query",
                (attention.tp_q_head_num, attention.head_dim),
                torch.bfloat16,
            ),
            (
                "diagnostic_attention_output",
                (attention.tp_q_head_num, attention.v_head_dim),
                torch.bfloat16,
            ),
            ("diagnostic_position", (), torch.int64),
            ("diagnostic_query_index", (1,), torch.int64),
        ):
            attention.register_buffer(
                name, torch.zeros(shape, dtype=dtype, device=device), persistent=False
            )
        attention.register_buffer(
            "diagnostic_attention_properties",
            torch.tensor(
                [attention.scaling, attention.logit_cap, attention.sliding_window_size],
                device=device,
                dtype=torch.float64,
            ),
            persistent=False,
        )
        attention.register_forward_hook(KVWriteHook())
        if capture_moe:
            hidden_size = layer.config.hidden_size
            for name, width in (
                ("diagnostic_attention_input", hidden_size),
                ("diagnostic_qkv_projection", layer.self_attn.qkv_proj.weight.shape[0]),
                ("diagnostic_attention_projection", hidden_size),
            ):
                attention.register_buffer(
                    name,
                    torch.zeros(width, dtype=torch.bfloat16, device=device),
                    persistent=False,
                )
            layer.self_attn.qkv_proj.register_forward_hook(
                ProjectionSnapshotHook(
                    attention.diagnostic_query_index,
                    attention.diagnostic_attention_input,
                    attention.diagnostic_qkv_projection,
                )
            )
            layer.self_attn.o_proj.register_forward_hook(
                ProjectionSnapshotHook(
                    attention.diagnostic_query_index,
                    None,
                    attention.diagnostic_attention_projection,
                )
            )
            layer.mlp.diagnostic_moe_query_index = attention.diagnostic_query_index
            for name, width, dtype in (
                ("diagnostic_moe_input", hidden_size, torch.bfloat16),
                (
                    "diagnostic_moe_router_logits",
                    layer.config.num_experts,
                    torch.float32,
                ),
                (
                    "diagnostic_moe_experts",
                    layer.config.num_experts_per_tok,
                    torch.int32,
                ),
                (
                    "diagnostic_moe_weights",
                    layer.config.num_experts_per_tok,
                    torch.float32,
                ),
                ("diagnostic_moe_shared_output", hidden_size, torch.bfloat16),
                ("diagnostic_moe_routed_output", hidden_size, torch.bfloat16),
                ("diagnostic_moe_output", hidden_size, torch.bfloat16),
            ):
                layer.mlp.register_buffer(
                    name,
                    torch.zeros(width, dtype=dtype, device=device),
                    persistent=False,
                )
        else:
            pass
