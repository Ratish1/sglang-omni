# SPDX-License-Identifier: Apache-2.0
"""K and V cache for CosyVoice3 causal Flow hops.

Flow is chunk causal, so a frame's K and V are final once its chunk is
complete. The cache keeps them per (Euler step, block, CFG lane) in SGLang's
paged pool, a stream's slots in two rows of SGLang's request table, and a hop
computes only its new frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    FA3_PAGE_SIZE,
    PackedDiT,
    PackedRows,
    chunk_segments,
    gather_rows,
    pack_rows,
    scatter_rows,
)

CFG_LANES = 2


@dataclass
class StreamHopCache:
    """One stream's request table rows, one per CFG lane, the frames they hold
    slots for and the frames whose K and V are written."""

    lanes: list[int]
    reserved: int = 0
    frames: int = 0


@dataclass
class CachedHop:
    """One hop's new frames in packed lane major order: where their K and V go,
    what each (row, chunk) query segment reads, and the row attention itself."""

    cache: "FlowHopCache"
    lanes: torch.Tensor
    lengths: torch.Tensor
    slots: torch.Tensor
    positions: torch.Tensor
    max_end: int
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    layer: int = 0

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """query, key, value: (1, new frames, heads * head_dim). Returns the
        same shape. Each call is the next (Euler step, block) pool layer."""
        cache = self.cache
        shape = (-1, cache.heads, cache.head_dim)
        cache.pool.set_kv_buffer(
            None,
            self.slots,
            key[0].reshape(shape),
            value[0].reshape(shape),
            layer_id_override=self.layer,
        )
        page_shape = (-1, FA3_PAGE_SIZE, cache.heads, cache.head_dim)
        out = flash_attn_with_kvcache(
            q=query[0].reshape(shape),
            k_cache=cache.pool.get_key_buffer(self.layer).view(page_shape),
            v_cache=cache.pool.get_value_buffer(self.layer).view(page_shape),
            cache_seqlens=self.cache_seqlens,
            page_table=self.page_table,
            cu_seqlens_q=self.cu_seqlens_q,
            max_seqlen_q=self.max_seqlen_q,
            causal=False,
        )
        self.layer += 1
        return out.reshape(1, -1, cache.heads * cache.head_dim)


class FlowHopCache:
    """The pool behind the cached hops. The byte budget covers the K and V
    slots, the request table rows and the conv position embedding tails."""

    def __init__(
        self,
        *,
        budget_bytes: int,
        steps: int,
        blocks: int,
        heads: int,
        head_dim: int,
        conv_context: int,
        conv_channels: int,
        chunk_size: int,
        max_frames: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> None:
        layers = steps * blocks
        self.bytes_per_slot = 2 * layers * heads * head_dim * dtype.itemsize
        # note(ratish): a lane that holds a row holds at least one chunk of slots.
        bytes_per_row = (
            steps * 2 * conv_context * conv_channels * dtype.itemsize
            + max_frames * torch.int32.itemsize
        )
        self.slots = (
            int(budget_bytes)
            * chunk_size
            // (self.bytes_per_slot * chunk_size + bytes_per_row)
        )
        row_count = self.slots // chunk_size
        if row_count < CFG_LANES:
            raise ValueError(
                f"flow_kv_cache_bytes={budget_bytes} holds {self.slots} frame "
                f"slots, below one chunk of {chunk_size} per CFG lane"
            )
        self.device = torch.device(device)
        self.blocks = blocks
        self.heads = heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.pool = MHATokenToKVPool(
            size=self.slots,
            page_size=FA3_PAGE_SIZE,
            dtype=dtype,
            head_num=heads,
            head_dim=head_dim,
            layer_num=layers,
            device=str(self.device),
            enable_memory_saver=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            self.slots, dtype, str(self.device), self.pool, need_sort=False
        )
        self.rows = ReqToTokenPool(
            size=row_count,
            max_context_len=max_frames,
            device=str(self.device),
            enable_memory_saver=False,
        )
        # note(ratish): the last conv_context inputs of both convs, per Euler
        # step and request table row; zeros are the causal conv's left padding.
        self.conv_tails = torch.zeros(
            steps,
            2,
            row_count + 1,
            conv_context,
            conv_channels,
            dtype=dtype,
            device=self.device,
        )
        self.fallback_hops = 0

    def open_stream(self) -> StreamHopCache | None:
        lanes = self.rows.alloc_rows(CFG_LANES)
        if lanes is None:
            return None
        self.conv_tails[:, :, lanes] = 0
        return StreamHopCache(lanes=lanes)

    def reserve(self, stream: StreamHopCache, end: int) -> bool:
        """Slots for the stream's frames [reserved, end) in both lanes."""
        count = end - stream.reserved
        fresh = self.allocator.alloc(count * CFG_LANES)
        if fresh is None:
            return False
        self.rows.req_to_token[stream.lanes, stream.reserved : end] = fresh.view(
            CFG_LANES, count
        ).to(torch.int32)
        stream.reserved = end
        return True

    def release(self, stream: StreamHopCache) -> None:
        held = self.rows.req_to_token[stream.lanes, : stream.reserved]
        self.allocator.free(held.reshape(-1).to(torch.int64))
        self.rows.free_rows(stream.lanes)

    def begin_hop(self, streams: Sequence[StreamHopCache]) -> CachedHop:
        """The hop over each stream's frames [frames, reserved), the rows of
        one CFG lane after the rows of the other."""
        spans = [(stream.frames, stream.reserved) for stream in streams] * CFG_LANES
        lanes = torch.tensor(
            [stream.lanes[lane] for lane in range(CFG_LANES) for stream in streams],
            device=self.device,
        )
        starts = torch.tensor([start for start, _ in spans], device=self.device)
        rows = pack_rows([end - start for start, end in spans], self.device)
        positions = rows.positions + starts[rows.row_ids]
        # note(ratish): a span starts on a chunk boundary, so its segments are
        # the segments of its length moved to its start.
        segment_rows, segment_ends, offsets = chunk_segments(
            rows.lengths, self.chunk_size
        )
        max_end = max(end for _, end in spans)
        as_int32 = {"dtype": torch.int32, "device": self.device}
        return CachedHop(
            cache=self,
            lanes=lanes,
            lengths=torch.tensor(rows.lengths, device=self.device),
            slots=self.rows.req_to_token[lanes[rows.row_ids], positions].to(
                torch.int64
            ),
            positions=positions,
            max_end=max_end,
            page_table=self.rows.req_to_token[
                lanes[torch.tensor(segment_rows, device=self.device)], :max_end
            ],
            cache_seqlens=torch.tensor(
                [
                    spans[row][0] + end
                    for row, end in zip(segment_rows, segment_ends, strict=True)
                ],
                **as_int32,
            ),
            cu_seqlens_q=torch.tensor(offsets, **as_int32),
            max_seqlen_q=max(end - start for start, end in pairwise(offsets)),
        )


class CachedDiT(PackedDiT):
    """PackedDiT over each row's new frames: attention reads the frames before
    them from the pool, the conv position embedding starts from the previous
    hop's last inputs, and RoPE is taken at absolute frame positions."""

    def __init__(
        self, dit: torch.nn.Module, cache: FlowHopCache, *, device: str | torch.device
    ) -> None:
        super().__init__(dit, device=device)
        self.cache = cache
        self.hop: CachedHop | None = None

    def row_attention(
        self, rows: PackedRows, *, streaming: bool, dtype: torch.dtype
    ) -> CachedHop:
        return self.hop

    def _rope(self, rows: PackedRows) -> tuple[torch.Tensor, Any]:
        freqs, scale = self.dit.rotary_embed.forward_from_seq_len(self.hop.max_end)
        freqs = freqs[:, self.hop.positions]
        if isinstance(scale, torch.Tensor):
            scale = scale[:, self.hop.positions]
        return freqs, scale

    def _conv_pos_embed(self, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
        hop = self.hop
        conv = self.dit.input_embed.conv_pos_embed
        context = conv.kernel_size - 1
        tails = self.cache.conv_tails[hop.layer // self.cache.blocks]
        first_tail, second_tail = tails[:, hop.lanes]
        first_in = torch.cat((first_tail, scatter_rows(h, rows, rows.width)), dim=1)
        first_out = conv.conv1(first_in.transpose(1, 2)).transpose(1, 2)
        second_in = torch.cat((second_tail, first_out), dim=1)
        second_out = conv.conv2(second_in.transpose(1, 2)).transpose(1, 2)
        # note(ratish): with the tail in front, a row's last context inputs
        # start at its own length.
        last = hop.lengths.unsqueeze(1) + torch.arange(context, device=h.device)
        last = last.unsqueeze(-1).expand(-1, -1, h.shape[2])
        tails[:, hop.lanes] = torch.stack(
            (first_in.gather(1, last), second_in.gather(1, last))
        )
        return gather_rows(second_out, rows)
