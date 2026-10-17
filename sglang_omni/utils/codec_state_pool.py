# SPDX-License-Identifier: Apache-2.0
"""In-place reads and writes of per-slot codec state: causal conv histories, transposed-conv
overlaps and the addresses of the K/V rings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.utils.custom_op import register_custom_op

from sglang_omni.utils.codec_state_arena import CodecSlotAllocator, CodecStateArenaStats
from sglang_omni.utils.incremental_codec import IncrementalCodecStateSpec

try:  # keep the module importable when Triton is unavailable
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

CHANNEL_BLOCK = 64

if HAS_TRITON:

    @triton.jit
    def history_exchange_kernel(
        padded_ptr,
        history_pool_ptr,
        slot_index_ptr,
        history_samples,
        new_samples,
        tail_samples,
        channels,
        padded_row_stride,
        padded_sample_stride,
        padded_channel_stride,
        pool_slot_stride,
        pool_sample_stride,
        pool_channel_stride,
        IS_COLD: tl.constexpr,
        IS_SHIFT: tl.constexpr,
        HISTORY_BLOCK: tl.constexpr,
        TAIL_BLOCK: tl.constexpr,
        CHANNEL_BLOCK: tl.constexpr,
    ):
        # note (ratish): one program owns a row's channel block over the whole history, so
        # every read of the slot's old history precedes every write of its new one.
        row = tl.program_id(0).to(tl.int64)
        channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
        channel_mask = channel < channels
        slot = tl.load(slot_index_ptr + row).to(tl.int64)
        sample = tl.arange(0, HISTORY_BLOCK)
        history_mask = (sample < history_samples)[:, None] & channel_mask[None, :]
        padded_base = (
            padded_ptr
            + row * padded_row_stride
            + channel[None, :] * padded_channel_stride
        )
        pool_base = (
            history_pool_ptr
            + slot * pool_slot_stride
            + channel[None, :] * pool_channel_stride
        )
        source = sample + new_samples
        from_new = (source >= history_samples)[:, None] & history_mask
        newest = tl.load(
            padded_base + source[:, None] * padded_sample_stride, mask=from_new, other=0
        )
        if IS_COLD:
            old = tl.zeros(
                [HISTORY_BLOCK, CHANNEL_BLOCK], dtype=padded_ptr.dtype.element_ty
            )
            new_history = newest
        else:
            old = tl.load(
                pool_base + sample[:, None] * pool_sample_stride,
                mask=history_mask,
                other=0,
            )
            if IS_SHIFT:
                kept = tl.load(
                    pool_base + source[:, None] * pool_sample_stride,
                    mask=history_mask & (source < history_samples)[:, None],
                    other=0,
                )
                new_history = tl.where(from_new, newest, kept)
                # note (ratish): with fewer new samples than the history, a kept sample is
                # read by one thread and overwritten by another.
                tl.debug_barrier()
            else:
                new_history = newest
        tl.store(
            padded_base + sample[:, None] * padded_sample_stride, old, mask=history_mask
        )
        tl.store(
            pool_base + sample[:, None] * pool_sample_stride,
            new_history,
            mask=history_mask,
        )
        tail = tl.arange(0, TAIL_BLOCK)
        tl.store(
            padded_base
            + (history_samples + new_samples + tail)[:, None] * padded_sample_stride,
            tl.zeros([TAIL_BLOCK, CHANNEL_BLOCK], dtype=padded_ptr.dtype.element_ty),
            mask=(tail < tail_samples)[:, None] & channel_mask[None, :],
        )

    @triton.jit
    def overlap_exchange_kernel(
        output_ptr,
        overlap_pool_ptr,
        slot_index_ptr,
        overlap_samples,
        emitted_samples,
        channels,
        output_row_stride,
        output_sample_stride,
        output_channel_stride,
        pool_slot_stride,
        pool_sample_stride,
        pool_channel_stride,
        IS_COLD: tl.constexpr,
        OVERLAP_BLOCK: tl.constexpr,
        CHANNEL_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
        channel_mask = channel < channels
        slot = tl.load(slot_index_ptr + row).to(tl.int64)
        sample = tl.arange(0, OVERLAP_BLOCK)
        mask = (sample < overlap_samples)[:, None] & channel_mask[None, :]
        output_base = (
            output_ptr
            + row * output_row_stride
            + channel[None, :] * output_channel_stride
        )
        pool_base = (
            overlap_pool_ptr
            + slot * pool_slot_stride
            + channel[None, :] * pool_channel_stride
        )
        tail = tl.load(
            output_base + (emitted_samples + sample)[:, None] * output_sample_stride,
            mask=mask,
            other=0,
        )
        if IS_COLD:
            pass
        else:
            previous = tl.load(
                pool_base + sample[:, None] * pool_sample_stride, mask=mask, other=0
            )
            head = tl.load(
                output_base + sample[:, None] * output_sample_stride, mask=mask, other=0
            )
            tl.store(
                output_base + sample[:, None] * output_sample_stride,
                (head.to(tl.float32) + previous.to(tl.float32)).to(head.dtype),
                mask=mask,
            )
        tl.store(pool_base + sample[:, None] * pool_sample_stride, tail, mask=mask)

else:
    pass


# note (ratish): opaque to torch.compile, which cannot trace the Triton launch
@register_custom_op(mutates_args=["padded", "history_pool"])
def launch_history_exchange(
    padded: torch.Tensor,
    history_pool: torch.Tensor,
    slot_index: torch.Tensor,
    new_samples: int,
    is_cold: bool,
) -> None:
    rows, padded_samples, channels = padded.shape
    history_samples = int(history_pool.shape[1])
    tail_samples = padded_samples - history_samples - new_samples
    with torch.get_device_module(padded.device).device(padded.device):
        history_exchange_kernel[(rows, triton.cdiv(channels, CHANNEL_BLOCK))](
            padded,
            history_pool,
            slot_index,
            history_samples,
            new_samples,
            tail_samples,
            channels,
            *padded.stride(),
            *history_pool.stride(),
            IS_COLD=is_cold,
            IS_SHIFT=new_samples < history_samples,
            HISTORY_BLOCK=triton.next_power_of_2(max(history_samples, 1)),
            TAIL_BLOCK=triton.next_power_of_2(max(tail_samples, 1)),
            CHANNEL_BLOCK=CHANNEL_BLOCK,
            num_warps=4,
        )


# note (ratish): opaque to torch.compile, which cannot trace the Triton launch
@register_custom_op(mutates_args=["output", "overlap_pool"])
def launch_overlap_exchange(
    output: torch.Tensor,
    overlap_pool: torch.Tensor,
    slot_index: torch.Tensor,
    emitted_samples: int,
    is_cold: bool,
) -> None:
    rows, _, channels = output.shape
    overlap_samples = int(overlap_pool.shape[1])
    with torch.get_device_module(output.device).device(output.device):
        overlap_exchange_kernel[(rows, triton.cdiv(channels, CHANNEL_BLOCK))](
            output,
            overlap_pool,
            slot_index,
            overlap_samples,
            emitted_samples,
            channels,
            *output.stride(),
            *overlap_pool.stride(),
            IS_COLD=is_cold,
            OVERLAP_BLOCK=triton.next_power_of_2(max(overlap_samples, 1)),
            CHANNEL_BLOCK=CHANNEL_BLOCK,
            num_warps=4,
        )


def history_exchange_reference(
    padded: torch.Tensor,
    history_pool: torch.Tensor,
    slot_index: torch.Tensor,
    new_samples: int,
    is_cold: bool,
) -> None:
    history_samples = int(history_pool.shape[1])
    if is_cold:
        padded[:, :history_samples].zero_()
    else:
        padded[:, :history_samples].copy_(history_pool.index_select(0, slot_index))
    padded[:, history_samples + new_samples :].zero_()
    history_pool.index_copy_(
        0, slot_index, padded[:, new_samples : new_samples + history_samples]
    )


def overlap_exchange_reference(
    output: torch.Tensor,
    overlap_pool: torch.Tensor,
    slot_index: torch.Tensor,
    emitted_samples: int,
    is_cold: bool,
) -> None:
    overlap_samples = int(overlap_pool.shape[1])
    if is_cold:
        pass
    else:
        output[:, :overlap_samples] += overlap_pool.index_select(0, slot_index)
    overlap_pool.index_copy_(
        0, slot_index, output[:, emitted_samples : emitted_samples + overlap_samples]
    )


def history_exchange(
    padded: torch.Tensor,
    history_pool: torch.Tensor,
    slot_index: torch.Tensor,
    *,
    new_samples: int,
    is_cold: bool,
) -> None:
    """Fill a causal conv's (rows, history + new + tail, channels) input around the new samples
    a producer already wrote: the slot's history in front (zeros for a cold slot), the dilation's
    zeros after; then keep the last history samples of history and new samples as the slot's
    history. history_pool is (slots, history, channels); slot_index names each row's slot.
    """
    if padded.device.type == "cuda":
        launch_history_exchange(padded, history_pool, slot_index, new_samples, is_cold)
    else:
        history_exchange_reference(
            padded, history_pool, slot_index, new_samples, is_cold
        )


def overlap_exchange(
    output: torch.Tensor,
    overlap_pool: torch.Tensor,
    slot_index: torch.Tensor,
    *,
    emitted_samples: int,
    is_cold: bool,
) -> None:
    """Add each slot's overlap into the front of a transposed conv's (rows, emitted + overlap,
    channels) output (not for a cold slot) and keep the samples past the emitted ones as the
    slot's overlap. Needs emitted_samples >= the overlap, so the two ranges never meet.
    """
    if output.device.type == "cuda":
        launch_overlap_exchange(
            output, overlap_pool, slot_index, emitted_samples, is_cold
        )
    else:
        overlap_exchange_reference(
            output, overlap_pool, slot_index, emitted_samples, is_cold
        )


@dataclass(frozen=True, kw_only=True)
class RingAddresses:
    """A decode's addresses in the per-slot K/V rings: the absolute positions of its new
    frames and the ring rows their K/V go to, and each row's attention window as ring rows in
    position order with its length."""

    frame_positions: torch.Tensor
    write_rows: torch.Tensor
    window_rows: torch.Tensor
    window_lengths: torch.Tensor


def ring_addresses(
    slot_index: torch.Tensor,
    start_positions: torch.Tensor,
    new_frames: int,
    ring_frames: int,
    retained_frames: int,
) -> RingAddresses:
    """Position p of a slot's stream lives at ring row slot * ring_frames + p % ring_frames. A
    row's window holds the retained_frames positions before its first new frame (fewer at the
    stream's start) and its new frames; ring_frames >= retained_frames + new_frames keeps them
    on distinct rows."""
    frame_offsets = torch.arange(new_frames, device=slot_index.device)
    frame_positions = start_positions[:, None] + frame_offsets
    slot_rows = slot_index[:, None] * ring_frames
    window_starts = (start_positions - retained_frames).clamp(min=0)
    window_positions = window_starts[:, None] + torch.arange(
        ring_frames, device=slot_index.device
    )
    return RingAddresses(
        frame_positions=frame_positions.reshape(-1),
        write_rows=(slot_rows + frame_positions % ring_frames).reshape(-1),
        window_rows=(slot_rows + window_positions % ring_frames).to(torch.int32),
        window_lengths=(start_positions - window_starts + new_frames).to(torch.int32),
    )


class CodecStatePool(CodecSlotAllocator):
    """Per-slot codec state kept where the decode's kernels read and write it: a history pool
    (slots, history samples, channels) per causal conv and an overlap pool (slots, overlap
    samples, channels) per transposed conv, a K/V ring of ring_frames positions per slot and
    layer, and each slot's frame position. Past the slots the streams take sit capture_slots
    scratch slots, one per row a graph capture decodes."""

    def __init__(
        self,
        spec: IncrementalCodecStateSpec,
        *,
        num_slots: int,
        capture_slots: int,
        ring_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(num_slots=num_slots, device=device)
        assert (
            ring_frames > spec.retained_context
        ), "a ring holds the window and new frames"
        total_slots = num_slots + capture_slots
        self.ring_frames = ring_frames
        self.retained_frames = spec.retained_context
        self.scratch_slots = tuple(range(num_slots, total_slots))
        self.histories = {
            key: torch.zeros(total_slots, length, channels, device=device, dtype=dtype)
            for key, channels, length in spec.conv_histories
            if length
        }
        self.overlaps = {
            key: torch.zeros(total_slots, length, channels, device=device, dtype=dtype)
            for key, channels, length in spec.transconv_overlaps
            if length
        }
        ring_width = spec.num_key_value_heads * spec.head_dim
        self.keys = [
            torch.zeros(
                total_slots * ring_frames, ring_width, device=device, dtype=dtype
            )
            for _ in range(spec.num_layers)
        ]
        self.values = [torch.zeros_like(ring) for ring in self.keys]
        self.frame_positions = torch.zeros(total_slots, device=device, dtype=torch.long)
        itemsize = torch.empty((), dtype=dtype).element_size()
        self.bytes_per_slot = itemsize * (
            sum(pool[0].numel() for pool in self.histories.values())
            + sum(pool[0].numel() for pool in self.overlaps.values())
            + 2 * spec.num_layers * ring_frames * ring_width
        )

    def zero_slots(self, slots: torch.Tensor) -> None:
        """Clear the slots' state, for a capture whose eager and replayed decodes must start
        from one state."""
        for pool in (*self.histories.values(), *self.overlaps.values()):
            pool.index_fill_(0, slots, 0)
        for ring in (*self.keys, *self.values):
            ring.view(-1, self.ring_frames, ring.shape[1]).index_fill_(0, slots, 0)
        self.frame_positions.index_fill_(0, slots, 0)

    def describe(self) -> CodecStateArenaStats:
        return {
            "slots": self.num_slots,
            "active_slots": self.active_slots(),
            "bytes_per_slot": self.bytes_per_slot,
            "total_bytes": self.bytes_per_slot * self.num_slots,
            "exhausted": self.exhausted_count,
        }
