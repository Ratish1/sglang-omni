# SPDX-License-Identifier: Apache-2.0
"""Apply a selected-row FP32 attention reference without changing native KV writes."""

from typing import Literal

import torch
from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd_normal
from sglang.kernels.ops.attention.kv_indices import create_flashinfer_kv_indices_triton
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend

from sglang_omni.scheduling.types import SchedulerRequest


class PrecisionAttentionBuffers:
    def __init__(
        self,
        batch_rows: int,
        table_tokens: int,
        query_heads: int,
        head_channels: int,
        device: torch.device,
    ) -> None:
        self.row_indices: torch.Tensor = torch.arange(
            batch_rows, device=device, dtype=torch.int32
        )
        self.kv_indices: torch.Tensor = torch.empty(
            batch_rows * table_tokens, device=device, dtype=torch.int32
        )
        self.split_counts: torch.Tensor = torch.ones(
            batch_rows, device=device, dtype=torch.int32
        )
        self.output: torch.Tensor = torch.empty(
            (batch_rows, query_heads, head_channels), device=device, dtype=torch.float32
        )
        self.partial_output: torch.Tensor = torch.empty(
            (batch_rows, query_heads, 1, head_channels),
            device=device,
            dtype=torch.float32,
        )
        self.partial_log_partition: torch.Tensor = torch.empty(
            (batch_rows, query_heads, 1), device=device, dtype=torch.float32
        )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        page_table: torch.Tensor,
        sequence_lengths: torch.Tensor,
        cumulative_sequence_lengths: torch.Tensor,
        refresh_indices: bool,
    ) -> torch.Tensor:
        batch_rows, query_heads, head_channels = query.shape
        assert query.dtype == keys.dtype == values.dtype == torch.bfloat16
        assert keys.shape == values.shape and keys.ndim == 3
        assert keys.shape[-1] == head_channels and query_heads % keys.shape[1] == 0
        assert self.output.shape == query.shape
        assert page_table.shape[0] == sequence_lengths.numel() == batch_rows
        assert cumulative_sequence_lengths.numel() == batch_rows + 1
        assert self.kv_indices.numel() >= page_table.numel()
        if refresh_indices:
            create_flashinfer_kv_indices_triton[(batch_rows,)](
                page_table,
                self.row_indices,
                sequence_lengths,
                cumulative_sequence_lengths,
                None,
                self.kv_indices,
                page_table.stride(0),
            )
        else:
            pass
        decode_attention_fwd_normal(
            q=query.float(),
            k_buffer=keys,
            v_buffer=values,
            o=self.output,
            kv_indptr=cumulative_sequence_lengths,
            kv_indices=self.kv_indices,
            attn_logits=self.partial_output,
            attn_lse=self.partial_log_partition,
            num_kv_splits=self.split_counts,
            max_kv_splits=1,
            sm_scale_withk=head_channels**-0.5,
            v_scale=1.0,
        )
        return self.output.to(query.dtype)


class NativeAttentionPrecisionTrial:
    def __init__(
        self,
        mode: Literal["control", "scalar"],
        request_ids: set[str],
        max_batch_rows: int,
        device: torch.device,
        attentions: list[RadixAttention],
    ) -> None:
        self.mode: Literal["control", "scalar"] = mode
        self.request_ids: set[str] = request_ids
        self.selected_rows: torch.Tensor = torch.zeros(
            max_batch_rows, device=device, dtype=torch.bool
        )
        self.buffers: dict[tuple[int, int], PrecisionAttentionBuffers] = {}
        assert request_ids and attentions
        for layer_index, attention in enumerate(attentions):
            assert attention.layer_id == layer_index
            attention.register_forward_hook(self.after_attention)

    def select(self, requests: list[SchedulerRequest]) -> None:
        assert len(requests) <= self.selected_rows.numel()
        self.selected_rows.zero_()
        if self.mode == "scalar":
            self.selected_rows[: len(requests)].copy_(
                torch.tensor(
                    [request.request_id in self.request_ids for request in requests],
                    device=self.selected_rows.device,
                    dtype=torch.bool,
                )
            )
        else:
            assert self.mode == "control"

    def after_attention(
        self,
        layer: RadixAttention,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, ForwardBatch],
        output: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value, forward_batch = inputs
        if not forward_batch.forward_mode.is_decode():
            return output
        else:
            pass
        backend = get_attn_backend()
        assert isinstance(backend, FlashAttentionBackend)
        assert backend.fa_impl_ver == 3 and backend.page_size == 1
        assert not backend.use_mla and forward_batch.spec_info is None
        assert layer.sliding_window_size == -1 and not layer.is_cross_attention
        assert layer.logit_cap == 0 and layer.scaling == layer.head_dim**-0.5
        metadata = backend.forward_metadata
        assert metadata.local_attn_metadata is None and metadata.max_seq_len_q == 1
        batch_rows = metadata.cache_seqlens_int32.numel()
        assert query.shape[0] == batch_rows <= self.selected_rows.numel()
        assert output.shape == (batch_rows, layer.tp_q_head_num * layer.head_dim)
        geometry = (batch_rows, metadata.page_table.shape[1])
        if geometry not in self.buffers:
            self.buffers[geometry] = PrecisionAttentionBuffers(
                batch_rows,
                geometry[1],
                layer.tp_q_head_num,
                layer.head_dim,
                query.device,
            )
        else:
            pass
        keys, values = backend.get_paged_mha_kv_cache(layer)
        assert keys.shape[1] == values.shape[1] == 1
        reference = (
            self.buffers[geometry]
            .forward(
                query.reshape(batch_rows, layer.tp_q_head_num, layer.head_dim),
                keys[:, 0],
                values[:, 0],
                metadata.page_table,
                metadata.cache_seqlens_int32,
                metadata.cu_seqlens_k,
                refresh_indices=layer.layer_id == 0,
            )
            .reshape_as(output)
        )
        return torch.where(self.selected_rows[:batch_rows, None], reference, output)
