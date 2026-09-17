# SPDX-License-Identifier: Apache-2.0
"""Per request K and V cache for CosyVoice3 causal Flow hops.

A hop recomputes the prompt and every earlier frame and then throws them away.
Flow is chunk causal, so a frame's K and V are final once its chunk is complete:
the cache keeps them per (Euler step, block, CFG lane) and a hop computes only
its new frames. Storage and the read path are SGLang's paged K/V pool and its
FA3 wrapper, one lane per CFG row, one pool layer per (Euler step, block).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    PackedRows,
    gather_rows,
    scatter_rows,
)

CACHE_DTYPE = torch.bfloat16
CFG_LANES = 2


class StreamHopCache:
    """One request's cached frames: the pool slots of each CFG lane, how many
    frames they cover, and the causal conv position embedding tails the next
    hop's first frames convolve against."""

    def __init__(self, cache: "FlowHopCache") -> None:
        self.cache = cache
        self.frames = 0
        self.slots: list[torch.Tensor] = [
            torch.empty(0, dtype=torch.int64, device=cache.device)
            for _ in range(CFG_LANES)
        ]
        self.tails: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def release(self) -> None:
        for lane, slots in enumerate(self.slots):
            if slots.numel():
                self.cache.allocator.free(slots)
            self.slots[lane] = torch.empty(
                0, dtype=torch.int64, device=self.cache.device
            )
        self.frames = 0
        self.tails.clear()


@dataclass(frozen=True)
class HopLayout:
    """The chunk segments one hop's query frames form, host arithmetic only.

    FA3 reads [0, cache_seqlens) for each query segment, so a hop is cut at the
    attention chunk boundaries it crosses: every frame of a segment sees the
    whole chunk it belongs to and every chunk before it, which is what the
    padded path's chunk mask gives it.
    """

    lanes: tuple[int, ...]
    cache_seqlens: tuple[int, ...]
    cu_seqlens_q: tuple[int, ...]
    max_seqlen_q: int


def hop_layout(spans: Sequence[tuple[int, int]], chunk: int) -> HopLayout:
    """spans: (first new frame, end frame) per lane row, in lane row order."""
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    lanes: list[int] = []
    cache_seqlens: list[int] = []
    cu_seqlens_q: list[int] = [0]
    for lane, (start, end) in enumerate(spans):
        if start < 0 or end <= start:
            raise ValueError(f"lane {lane} span {(start, end)} is not a forward span")
        frame = start
        while frame < end:
            segment_end = min((frame // chunk + 1) * chunk, end)
            lanes.append(lane)
            cache_seqlens.append(segment_end)
            cu_seqlens_q.append(cu_seqlens_q[-1] + segment_end - frame)
            frame = segment_end
    return HopLayout(
        lanes=tuple(lanes),
        cache_seqlens=tuple(cache_seqlens),
        cu_seqlens_q=tuple(cu_seqlens_q),
        max_seqlen_q=max(
            cu_seqlens_q[index + 1] - cu_seqlens_q[index]
            for index in range(len(cache_seqlens))
        ),
    )


@dataclass
class CachedHopCall:
    """Where one hop's new frames live in the pool, which slots each query
    segment may read, and each row's conv tails. Built once per call and shared
    by all pool layers."""

    pool: Any
    heads: int
    head_dim: int
    entries: list[tuple[StreamHopCache, int]]
    lengths: torch.Tensor
    slots: torch.Tensor
    positions: torch.Tensor
    max_frames: int
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int

    def conv_tails(
        self, step: int, like: torch.Tensor, context: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = like.new_zeros(context, like.shape[2])
        pairs = [entry.tails.get((lane, step)) for entry, lane in self.entries]
        return (
            torch.stack([zeros if pair is None else pair[0] for pair in pairs]),
            torch.stack([zeros if pair is None else pair[1] for pair in pairs]),
        )

    def store_conv_tails(
        self, step: int, conv1_in: torch.Tensor, conv2_in: torch.Tensor, context: int
    ) -> None:
        index = self.lengths.unsqueeze(1) + torch.arange(
            context, device=conv1_in.device
        ).unsqueeze(0)
        index = index.unsqueeze(-1).expand(-1, -1, conv1_in.shape[2])
        tails1 = torch.gather(conv1_in, 1, index)
        tails2 = torch.gather(conv2_in, 1, index)
        for row, (entry, lane) in enumerate(self.entries):
            entry.tails[(lane, step)] = (tails1[row], tails2[row])


class CachedRowAttention:
    """Attention for the new frames against the pool: the row's new K and V go
    to its fresh slots, then one FA3 call reads [0, chunk end) for every
    (lane row, chunk) query segment through the page table."""

    def __init__(self, call: CachedHopCall) -> None:
        self.call = call
        self.layer = 0

    def __call__(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        call = self.call
        shape = (-1, call.heads, call.head_dim)
        call.pool.set_kv_buffer(
            None,
            call.slots,
            key[0].reshape(shape).to(CACHE_DTYPE).contiguous(),
            value[0].reshape(shape).to(CACHE_DTYPE).contiguous(),
            layer_id_override=self.layer,
        )
        out = flash_attn_with_kvcache(
            query[0].reshape(shape).to(CACHE_DTYPE),
            call.pool.get_key_buffer(self.layer).view(-1, 1, call.heads, call.head_dim),
            call.pool.get_value_buffer(self.layer).view(
                -1, 1, call.heads, call.head_dim
            ),
            cache_seqlens=call.cache_seqlens,
            page_table=call.page_table,
            cu_seqlens_q=call.cu_seqlens_q,
            max_seqlen_q=call.max_seqlen_q,
            causal=False,
        )
        self.layer += 1
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(1, -1, call.heads * call.head_dim).to(query.dtype)


class CachedDiT(PackedDiT):
    """PackedDiT over the new frames only: the same modules in the same order,
    attention reading the pool, the conv position embedding starting from the
    previous hop's tails, and RoPE at absolute frame positions."""

    def __init__(self, dit: torch.nn.Module, call: CachedHopCall) -> None:
        super().__init__(dit)
        self.call = call
        self.step = 0

    def row_attention(self, rows: PackedRows, *, streaming: bool) -> CachedRowAttention:
        return CachedRowAttention(self.call)

    def _rope(self, rows: PackedRows) -> tuple[torch.Tensor, Any]:
        freqs, scale = self.dit.rotary_embed.forward_from_seq_len(self.call.max_frames)
        freqs = freqs[:, self.call.positions]
        if isinstance(scale, torch.Tensor):
            scale = scale[:, self.call.positions]
        return freqs, scale

    def _conv_pos_embed(self, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
        conv = self.dit.input_embed.conv_pos_embed
        context = conv.kernel_size - 1
        padded = scatter_rows(h, rows, rows.width)
        tail_h, tail_conv1 = self.call.conv_tails(self.step, padded, context)
        conv1_in = torch.cat((tail_h, padded), dim=1)
        conv1_out = conv.conv1(conv1_in.transpose(1, 2)).transpose(1, 2)
        conv2_in = torch.cat((tail_conv1, conv1_out), dim=1)
        conv2_out = conv.conv2(conv2_in.transpose(1, 2)).transpose(1, 2)
        self.call.store_conv_tails(self.step, conv1_in, conv2_in, context)
        self.step += 1
        return gather_rows(conv2_out, rows)


@dataclass
class FlowHopCacheStats:
    calls: int = 0
    cached_frames: int = 0
    fallback_hops: int = 0


class FlowHopCache:
    """The pool behind the cached hops and the streams that hold slots in it."""

    def __init__(
        self,
        *,
        budget_bytes: int,
        layers: int,
        heads: int,
        head_dim: int,
        chunk: int,
        device: str | torch.device,
    ) -> None:
        from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        self.bytes_per_frame = (
            2 * layers * heads * head_dim * CACHE_DTYPE.itemsize
        )
        slots = int(budget_bytes) // self.bytes_per_frame
        if slots <= 0:
            raise ValueError(
                f"flow_kv_cache_bytes must cover at least one frame "
                f"({self.bytes_per_frame} bytes), got {budget_bytes}"
            )
        self.device = torch.device(device)
        self.slots = slots
        self.chunk = int(chunk)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.pool = MHATokenToKVPool(
            size=slots,
            page_size=1,
            dtype=CACHE_DTYPE,
            head_num=heads,
            head_dim=head_dim,
            layer_num=layers,
            device=str(self.device),
            enable_memory_saver=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            slots, CACHE_DTYPE, str(self.device), self.pool, need_sort=False
        )
        self.stats = FlowHopCacheStats()

    def open_stream(self) -> StreamHopCache:
        return StreamHopCache(self)

    @property
    def free_slots(self) -> int:
        return int(self.allocator.available_size())

    def begin_call(
        self, rows: Sequence[tuple[StreamHopCache, int, int]]
    ) -> CachedHopCall:
        """rows: (stream cache, first new frame, end frame) in row order. A
        row's CFG twin is a separate lane with its own slots, and the lanes are
        laid out lane major so one packed call covers both."""
        for entry, start, end in rows:
            if entry.frames != start:
                raise RuntimeError(
                    f"stream holds {entry.frames} cached frames, "
                    f"the hop starts at {start}"
                )
            if end <= start:
                raise RuntimeError(f"hop span {(start, end)} is not a forward span")
        max_frames = max(end for _, _, end in rows)
        entries: list[tuple[StreamHopCache, int]] = []
        lengths: list[int] = []
        fresh_slots: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for lane in range(CFG_LANES):
            for entry, start, end in rows:
                fresh = self.allocator.alloc(end - start)
                if fresh is None:
                    raise RuntimeError(
                        f"Flow K/V pool exhausted: {self.allocator.available_size()} "
                        f"slots free, {end - start} needed"
                    )
                entry.slots[lane] = torch.cat((entry.slots[lane], fresh))
                entries.append((entry, lane))
                lengths.append(end - start)
                fresh_slots.append(fresh)
                positions.append(
                    torch.arange(start, end, device=self.device, dtype=torch.int64)
                )
        for entry, _, end in rows:
            entry.frames = end
        layout = hop_layout(
            [(start, end) for _ in range(CFG_LANES) for _, start, end in rows],
            self.chunk,
        )
        lane_table = torch.zeros(
            len(entries), max_frames, dtype=torch.int32, device=self.device
        )
        for row, (entry, lane) in enumerate(entries):
            slots = entry.slots[lane]
            lane_table[row, : slots.numel()] = slots.to(torch.int32)
        as_int32 = {"dtype": torch.int32, "device": self.device}
        self.stats.calls += 1
        self.stats.cached_frames += sum(lengths)
        return CachedHopCall(
            pool=self.pool,
            heads=self.heads,
            head_dim=self.head_dim,
            entries=entries,
            lengths=torch.tensor(lengths, dtype=torch.int64, device=self.device),
            slots=torch.cat(fresh_slots),
            positions=torch.cat(positions),
            max_frames=max_frames,
            page_table=lane_table[
                torch.tensor(layout.lanes, dtype=torch.int64, device=self.device)
            ],
            cache_seqlens=torch.tensor(layout.cache_seqlens, **as_int32),
            cu_seqlens_q=torch.tensor(layout.cu_seqlens_q, **as_int32),
            max_seqlen_q=layout.max_seqlen_q,
        )
