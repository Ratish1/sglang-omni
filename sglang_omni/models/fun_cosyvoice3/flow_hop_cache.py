# SPDX-License-Identifier: Apache-2.0
"""K and V cache for CosyVoice3 causal Flow hops.

Flow is chunk causal, so a frame's K and V are final once its chunk is
complete. The cache keeps them per (Euler step, block, CFG lane) in SGLang's
paged pool, a stream's slots in two rows of SGLang's request table, and a hop
computes only its new frames. One Euler step over those frames is captured as
a breakable CUDA graph per step size, the attention and the conv position
embedding left eager between its segments.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)

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
# Padding over the cached steps of a c16 SeedTTS stream run: 3.2 % of the real
# frames with 21 sizes; 4 gives 6.1 % with 13, 16 gives 2.0 % with 33.
STEP_SIZES_PER_DOUBLING = 8


def step_sizes(unit: int, limit: int) -> tuple[int, ...]:
    """Packed step sizes to capture, in frames over both CFG lanes: multiples
    of unit whose spacing doubles every STEP_SIZES_PER_DOUBLING sizes, then the
    largest multiple the limit holds."""
    top = limit // unit * unit
    sizes: list[int] = []
    size = spacing = unit
    while size < top:
        sizes.append(size)
        if size >= STEP_SIZES_PER_DOUBLING * spacing:
            spacing *= 2
        size += spacing
    return (*sizes, top)


def pad_frames(packed: torch.Tensor, frames: int) -> torch.Tensor:
    """(1, total, channels) -> (1, frames, channels), zeros past total."""
    if packed.shape[1] == frames:
        return packed
    return F.pad(packed, (0, 0, 0, frames - packed.shape[1]))


@dataclass
class StreamHopCache:
    """One stream's request table rows, one per CFG lane, the frames they hold
    slots for and the frames whose K and V are written."""

    lanes: list[int]
    reserved: int = 0
    frames: int = 0


@dataclass
class CachedHop:
    """One hop's new frames in packed lane major order: the (row, chunk) query
    segments, the frames each one's lane holds before it, and the Euler step
    and block the solve is at."""

    rows: PackedRows
    lanes: torch.Tensor
    lengths: torch.Tensor
    positions: torch.Tensor
    max_end: int
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    step: int = 0
    block: int = 0


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
        page_shape = (-1, FA3_PAGE_SIZE, heads, head_dim)
        self.pages = [
            (
                self.pool.get_key_buffer(layer).view(page_shape),
                self.pool.get_value_buffer(layer).view(page_shape),
            )
            for layer in range(layers)
        ]
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
            rows=rows,
            lanes=lanes,
            lengths=torch.tensor(rows.lengths, device=self.device),
            positions=positions,
            max_end=max_end,
            page_table=self.rows.req_to_token[
                lanes[torch.tensor(segment_rows, device=self.device)], :max_end
            ],
            cache_seqlens=torch.tensor(
                [
                    spans[row][0] + end - (after - before)
                    for row, end, (before, after) in zip(
                        segment_rows, segment_ends, pairwise(offsets), strict=True
                    )
                ],
                **as_int32,
            ),
            cu_seqlens_q=torch.tensor(offsets, **as_int32),
            max_seqlen_q=max(end - start for start, end in pairwise(offsets)),
        )


class CachedDiT(PackedDiT):
    """PackedDiT over each row's new frames: attention reads the frames before
    them from the pool, the conv position embedding starts from the previous
    hop's last inputs, and RoPE is taken at absolute frame positions. With
    capture_steps an Euler step replays from the breakable CUDA graph of its
    step size; attention, the conv and RoPE are its eager breaks."""

    def __init__(
        self,
        dit: torch.nn.Module,
        cache: FlowHopCache,
        *,
        device: str | torch.device,
        capture_steps: bool,
    ) -> None:
        super().__init__(dit, device=device)
        self.cache = cache
        self.hop: CachedHop | None = None
        self.hop_rope: tuple[torch.Tensor, float] | None = None
        self.step_frames = 0
        self.run_step: Callable[..., torch.Tensor] = self.step
        self.sizes: tuple[int, ...] = ()
        self.graphs: BaseBreakableCudaGraphRunner | None = None
        # note(ratish): the size the startup warmup is capturing; serving
        # replays and never captures.
        self.capture_size: int | None = None
        if capture_steps:
            # note(ratish): a cached step holds a slot per frame, so the pool
            # bounds it; no capture is evicted, warmup alone decides the set.
            self.sizes = step_sizes(CFG_LANES * cache.chunk_size, cache.slots)
            self.graphs = BaseBreakableCudaGraphRunner(self.step, cache.device)
            self.graphs.max_entries = 0

    def begin_hop(self, streams: Sequence[StreamHopCache]) -> None:
        """The hop over the streams' new frames: its segments, its RoPE at
        absolute positions, and the captured step size it runs at, if any."""
        hop = self.cache.begin_hop(streams)
        index = bisect_left(self.sizes, hop.rows.total)
        if self.capture_size is not None:
            self.step_frames, self.run_step = self.capture_size, self.graphs
        elif index < len(self.sizes):
            self.step_frames, self.run_step = self.sizes[index], self.graphs
        else:
            self.step_frames, self.run_step = hop.rows.total, self.step
        freqs, scale = self.dit.rotary_embed.forward_from_seq_len(hop.max_end)
        assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
        self.hop = hop
        self.hop_rope = pad_frames(freqs[:, hop.positions], self.step_frames), scale

    def row_attention(
        self, rows: PackedRows, *, streaming: bool, dtype: torch.dtype
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        return self.attend

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        rows: PackedRows,
        attention: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """One Euler step at the hop's step size, the frames past the hop's
        own zero. The solver's rows and attention are the hop's, which the
        breaks read from the hop itself."""
        total = x.shape[1]
        inputs = {
            "x": pad_frames(x, self.step_frames),
            "mu": pad_frames(mu, self.step_frames),
            "spks": pad_frames(spks, self.step_frames),
            "cond": pad_frames(cond, self.step_frames),
            "t": t,
        }
        if self.capture_size is not None:
            # note(ratish): a weight cast that autocast caches dies with its
            # context, so the captured casts must be the graph's own.
            device_type = x.device.type
            with torch.autocast(
                device_type,
                dtype=torch.get_autocast_dtype(device_type),
                cache_enabled=False,
            ):
                self.graphs.capture(**inputs)
        out = self.run_step(**inputs)[:, :total]
        self.hop.step += 1
        return out

    def step(
        self,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """x, mu, spks, cond: (1, step frames, channels). A captured step keeps
        only tensor addresses, so everything else comes from the hop."""
        return super().forward(x, mu, spks, cond, t, self.hop.rows, self.attend)

    @eager_on_graph(True)
    def attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """query, key, value: (1, step frames, heads * head_dim), the hop's
        frames first. Returns the same shape. Each call is the step's next
        block, whose pool layer is step * blocks + block."""
        hop, cache = self.hop, self.cache
        total = hop.rows.total
        shape = (-1, cache.heads, cache.head_dim)
        k_pages, v_pages = cache.pages[hop.step * cache.blocks + hop.block]
        # note(ratish): FA3 appends each segment's K and V at cache_seqlens
        # through the page table, then attends; a row's later chunk reads what
        # its earlier chunk appended in the same call.
        out = flash_attn_with_kvcache(
            q=query[0, :total].reshape(shape),
            k_cache=k_pages,
            v_cache=v_pages,
            k=key[0, :total].reshape(shape),
            v=value[0, :total].reshape(shape),
            cache_seqlens=hop.cache_seqlens,
            page_table=hop.page_table,
            cu_seqlens_q=hop.cu_seqlens_q,
            cu_seqlens_k_new=hop.cu_seqlens_q,
            max_seqlen_q=hop.max_seqlen_q,
            causal=False,
        )
        hop.block += 1
        return pad_frames(out.reshape(1, total, -1), query.shape[1])

    # note(ratish): rows is the hook's contract; the breaks below replay with
    # the arguments of the capture, so they read this call's rows from the hop.
    @eager_on_graph(True)
    def rope(self, rows: PackedRows) -> tuple[torch.Tensor, float]:
        return self.hop_rope

    @eager_on_graph(True)
    def conv_pos_embed(self, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
        hop = self.hop
        rows = hop.rows
        # note(ratish): the conv opens every run of a step, the repeated runs
        # of a capture too, so the step's block count restarts here.
        hop.block = 0
        conv = self.dit.input_embed.conv_pos_embed
        context = conv.kernel_size - 1
        tails = self.cache.conv_tails[hop.step]
        first_tail, second_tail = tails[:, hop.lanes]
        first_in = torch.cat(
            (first_tail, scatter_rows(h[:, : rows.total], rows, rows.width)), dim=1
        )
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
        return pad_frames(gather_rows(second_out, rows), h.shape[1])
