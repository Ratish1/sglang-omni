"""Retain serving attention page metadata beside captured cache histories."""

from __future__ import annotations

import torch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def capture_attention_layout(
    forward_batch: ForwardBatch,
    key_buffers: list[torch.Tensor],
    value_buffers: list[torch.Tensor],
    request_slots: torch.Tensor,
    row: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    backend = forward_batch.attn_backend
    assert isinstance(backend, FlashAttentionBackend)
    metadata = backend.forward_metadata
    page_count = (sequence_length + backend.page_size - 1) // backend.page_size
    return {
        "attention_page_ids": metadata.page_table[row, :page_count].cpu().clone(),
        "attention_cache_length": metadata.cache_seqlens_int32[row].cpu().clone(),
        "attention_page_size": torch.tensor(backend.page_size),
        "attention_version": torch.tensor(backend.fa_impl_ver),
        "attention_backend_request_pointer": torch.tensor(
            backend.req_to_token.data_ptr()
        ),
        "attention_recorded_request_pointer": torch.tensor(request_slots.data_ptr()),
        "attention_backend_key_pointers": torch.tensor(
            [
                backend.token_to_kv_pool.get_key_buffer(layer_index).data_ptr()
                for layer_index in range(len(key_buffers))
            ]
        ),
        "attention_recorded_key_pointers": torch.tensor(
            [buffer.data_ptr() for buffer in key_buffers]
        ),
        "attention_backend_value_pointers": torch.tensor(
            [
                backend.token_to_kv_pool.get_value_buffer(layer_index).data_ptr()
                for layer_index in range(len(value_buffers))
            ]
        ),
        "attention_recorded_value_pointers": torch.tensor(
            [buffer.data_ptr() for buffer in value_buffers]
        ),
        "attention_uses_mla": torch.tensor(backend.use_mla),
        "attention_has_sliding_window": torch.tensor(backend.has_swa),
        "attention_uses_sliding_window_pool": torch.tensor(
            backend.use_sliding_window_kv_pool
        ),
        "attention_has_local_builder": torch.tensor(
            backend.local_attn_builder is not None
        ),
        "attention_uses_prefill_aware_swa": torch.tensor(backend.is_prefill_aware_swa),
        "attention_uses_unified_read": torch.tensor(
            backend.kv_index_translator.is_translating
        ),
        "attention_uses_sharded_pool": torch.tensor(
            backend._kv_shard_pool
            is not None  # noqa: leading-underscore - pinned backend field
        ),
        "attention_skips_prefill_cache": torch.tensor(backend.fa_skip_kv_cache),
    }
