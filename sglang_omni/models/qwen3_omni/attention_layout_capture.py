"""Retain native attention scheduling and the live first-layer KV read footprint."""

from __future__ import annotations

import torch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention


def capture_attention_layout(
    backend: FlashAttentionBackend,
    layer: RadixAttention,
    key_buffers: list[torch.Tensor],
    value_buffers: list[torch.Tensor],
    request_slots: torch.Tensor,
    row: int,
    sequence_length: int,
    active_request_count: int,
) -> dict[str, torch.Tensor]:
    assert isinstance(backend, FlashAttentionBackend)
    assert backend.page_size == 1
    assert layer.attn_type == AttentionType.DECODER and not layer.is_cross_attention
    assert layer.k_scale is None and layer.v_scale is None
    metadata = backend.forward_metadata
    key_cache, value_cache = backend.get_paged_mha_kv_cache(layer)
    assert key_cache.data_ptr() == key_buffers[0].data_ptr()
    assert value_cache.data_ptr() == value_buffers[0].data_ptr()
    cache_lengths = metadata.cache_seqlens_int32
    query_lengths = metadata.cu_seqlens_q.diff()
    page_positions = torch.arange(
        metadata.page_table.shape[1], device=cache_lengths.device
    )
    live_pages = (page_positions[None] < cache_lengths[:, None]) & (
        query_lengths[:, None] > 0
    )
    locations = torch.unique(metadata.page_table[live_pages]).long()
    assert bool((locations >= 0).all())
    parameters: dict[str, int] = {
        "attention_max_query_tokens": metadata.max_seq_len_q,
        "attention_max_key_tokens": metadata.max_seq_len_k,
        "attention_split_count": backend.num_splits,
        "attention_decode_split_count": backend.decode_num_splits,
        "attention_query_row": row,
        "attention_active_request_count": active_request_count,
        "attention_page_size_tokens": backend.page_size,
        "attention_version": backend.fa_impl_ver,
        "attention_query_head_count": layer.tp_q_head_num,
        "attention_key_head_count": layer.tp_k_head_num,
        "attention_query_head_dimensions": layer.head_dim,
        "attention_value_head_dimensions": layer.v_head_dim,
    }
    return {
        "attention_page_ids": metadata.page_table[row, :sequence_length].cpu().clone(),
        "attention_page_table": metadata.page_table.cpu().clone(),
        "attention_cache_length": cache_lengths[row].cpu().clone(),
        "attention_cache_lengths": cache_lengths.cpu().clone(),
        "attention_cumulative_query_lengths": metadata.cu_seqlens_q.cpu().clone(),
        "attention_scheduler_metadata_after_forward": (
            metadata.scheduler_metadata.cpu().clone()
            if metadata.scheduler_metadata is not None
            else torch.empty(0, dtype=torch.int32, device="cpu")
        ),
        **{
            name: torch.tensor(value, device="cpu")
            for name, value in parameters.items()
        },
        "attention_scaling": torch.tensor(
            layer.scaling, dtype=torch.float64, device="cpu"
        ),
        "attention_logit_cap": torch.tensor(
            layer.logit_cap, dtype=torch.float64, device="cpu"
        ),
        "attention_key_buffer_shape": torch.tensor(key_buffers[0].shape, device="cpu"),
        "attention_key_cache_shape": torch.tensor(key_cache.shape, device="cpu"),
        "attention_key_cache_strides": torch.tensor(key_cache.stride(), device="cpu"),
        "attention_value_cache_shape": torch.tensor(value_cache.shape, device="cpu"),
        "attention_value_cache_strides": torch.tensor(
            value_cache.stride(), device="cpu"
        ),
        "attention_live_locations": locations.cpu(),
        "attention_live_layer0_keys": key_buffers[0].index_select(0, locations).cpu(),
        "attention_live_layer0_values": value_buffers[0]
        .index_select(0, locations)
        .cpu(),
        "attention_backend_request_pointer": torch.tensor(
            backend.req_to_token.data_ptr(), device="cpu"
        ),
        "attention_recorded_request_pointer": torch.tensor(
            request_slots.data_ptr(), device="cpu"
        ),
        "attention_backend_key_pointers": torch.tensor(
            [
                backend.token_to_kv_pool.get_key_buffer(index).data_ptr()
                for index in range(len(key_buffers))
            ],
            device="cpu",
        ),
        "attention_recorded_key_pointers": torch.tensor(
            [buffer.data_ptr() for buffer in key_buffers], device="cpu"
        ),
        "attention_backend_value_pointers": torch.tensor(
            [
                backend.token_to_kv_pool.get_value_buffer(index).data_ptr()
                for index in range(len(value_buffers))
            ],
            device="cpu",
        ),
        "attention_recorded_value_pointers": torch.tensor(
            [buffer.data_ptr() for buffer in value_buffers], device="cpu"
        ),
    }
