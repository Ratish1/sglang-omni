"""Capture KV writes and attention operands inside model CUDA graphs."""

from __future__ import annotations

import os

import torch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from torch import nn


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


def install_kv_write_observers(layers: nn.ModuleList) -> None:
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
