# SPDX-License-Identifier: Apache-2.0
"""Per-stream codec decoder state held by slot: the state's layout, the slot allocator, and the
pool whose histories, overlaps and K/V rings a decode's kernels read and write in place."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from sglang_omni.utils.cuda_staging import PinnedTransferSlot
from sglang_omni.utils.device import supports_device_streams

# A thread has at most two cohorts between launch and resolve, each staging its slot index
# twice, so a ring of four rewrites no host buffer whose copy is still queued.
STAGING_RING_DEPTH = 4

StagingName = Literal["index", "valid_frames"]


@dataclass(frozen=True)
class CodecStateSpec:
    """One stream's incremental codec state: (key, channels, length) conv histories and
    transposed-conv overlaps in decode order, zero lengths kept, and its K/V shape."""

    conv_histories: tuple[tuple[str, int, int], ...]
    transconv_overlaps: tuple[tuple[str, int, int], ...]
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    retained_context: int

    def bytes_per_stream(self, dtype: torch.dtype) -> int:
        elements = sum(channels * length for _, channels, length in self.conv_histories)
        elements += sum(
            channels * length for _, channels, length in self.transconv_overlaps
        )
        elements += (
            2
            * self.num_layers
            * self.num_key_value_heads
            * self.retained_context
            * self.head_dim
        )
        return elements * dtype.itemsize


class CausalConv(Protocol):
    """A stride-1 causal conv: the conv and the history length it pads on the left."""

    conv: torch.nn.Conv1d
    padding: int


class CausalTransConv(Protocol):
    """A causal transposed conv: the conv and the samples its whole-sequence forward trims
    from the start and the end."""

    conv: torch.nn.ConvTranspose1d
    left_pad: int
    right_pad: int


class CodecConvStack(Protocol):
    """The conv stack after the transformer: upsample stages of a transposed conv and a
    ConvNeXt block, then decoder.0, decoder blocks (activation, transposed conv, residual
    units), a final activation and a final conv."""

    upsample: torch.nn.ModuleList
    decoder: torch.nn.ModuleList


def conv_stack_state_layout(
    codec: CodecConvStack,
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """The conv stack's (key, channels, length) conv histories and transposed-conv overlaps,
    in the order its decode visits them; Qwen3-Omni's code2wav and Qwen3-TTS's speech
    tokenizer decoder share this stack."""
    conv_histories: list[tuple[str, int, int]] = []
    transconv_overlaps: list[tuple[str, int, int]] = []

    def add_conv(module: CausalConv, state_key: str) -> None:
        conv_histories.append(
            (state_key, int(module.conv.in_channels), int(module.padding))
        )

    def add_transconv(module: CausalTransConv, state_key: str) -> None:
        transconv_overlaps.append(
            (state_key, int(module.conv.out_channels), int(module.right_pad))
        )

    for stage_index, blocks in enumerate(codec.upsample):
        add_transconv(blocks[0], f"upsample.{stage_index}.transconv")
        add_conv(blocks[1].dwconv, f"upsample.{stage_index}.convnext.dwconv")
    add_conv(codec.decoder[0], "decoder.0")
    for block_index, decoder_block in enumerate(codec.decoder[1:-2], start=1):
        add_transconv(decoder_block.block[1], f"decoder.{block_index}.transconv")
        for residual_index, residual in enumerate(decoder_block.block[2:]):
            residual_key = f"decoder.{block_index}.residual.{residual_index}"
            add_conv(residual.conv1, f"{residual_key}.conv1")
            add_conv(residual.conv2, f"{residual_key}.conv2")
    add_conv(codec.decoder[-1], "decoder.final")
    return conv_histories, transconv_overlaps


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
    """Position p of a slot's stream lives at ring row slot * ring_frames + p % ring_frames; a
    row's window is the retained_frames positions before its first new frame and its new
    frames, on distinct rows while ring_frames >= retained_frames + new_frames."""
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
        window_rows=slot_rows + window_positions % ring_frames,
        window_lengths=start_positions - window_starts + new_frames,
    )


@dataclass(kw_only=True)
class StagingRing:
    """Pinned host buffers and the device buffers they copy into, taken in turn."""

    entries: list[tuple[PinnedTransferSlot, torch.Tensor]]
    turn: int = 0


class ThreadStaging(threading.local):
    """A thread's staging rings by name: a ring's device buffers are reused on its thread's
    stream only, which orders every read of them before the next copy."""

    def __init__(self) -> None:
        self.rings: dict[StagingName, StagingRing] = {}


class CodecSlotAllocator:
    """Bounded, reusable slots for per-stream codec state: a slot is handed out after every
    decode its previous owner queued, and slot ids reach the device without a host sync.
    """

    def __init__(self, *, num_slots: int, device: torch.device) -> None:
        if num_slots <= 0:
            raise ValueError("codec state needs at least one slot")
        else:
            pass
        self.device = torch.device(device)
        self.device_module = torch.get_device_module(self.device)
        self.num_slots = num_slots
        self.exhausted_count = 0
        self.lock = threading.Lock()
        self.staging = ThreadStaging()
        self.release_events: dict[int, torch.Event] = {}
        self.free: list[int] = list(reversed(range(num_slots)))
        self.retired: set[int] = set()

    def active_slots(self) -> int:
        with self.lock:
            return self.num_slots - len(self.free) - len(self.retired)

    def acquire(self) -> int | None:
        """Take a slot once the current stream is ordered after its previous owner's decodes,
        or None when every slot is taken."""
        with self.lock:
            if not self.free:
                self.exhausted_count += 1
                return None
            else:
                pass
            slot = self.free.pop()
            release_event = self.release_events.pop(slot, None)
        if release_event is not None:
            self.device_module.current_stream(self.device).wait_event(release_event)
        else:
            pass
        return slot

    def release(self, slot: int) -> None:
        release_event = None
        if supports_device_streams(self.device):
            release_event = self.device_module.Event()
            release_event.record(self.device_module.current_stream(self.device))
        else:
            pass
        with self.lock:
            if slot in self.retired:
                return
            else:
                pass
            assert slot not in self.free, f"codec state slot {slot} was released twice"
            if release_event is not None:
                self.release_events[slot] = release_event
            else:
                pass
            self.free.append(slot)

    def retire(self, slot: int) -> None:
        """Withdraw a slot for the life of the process.

        Note (Qihao Liu): used when a decode that touched the slot could not be
        proven complete, so its memory must never be handed to later work.
        """
        with self.lock:
            self.retired.add(slot)
            self.release_events.pop(slot, None)
            if slot in self.free:
                self.free.remove(slot)
            else:
                pass

    def staged(self, name: StagingName, values: Sequence[int]) -> torch.Tensor:
        """The values on the device, copied from the thread's next pinned buffer for name."""
        value_count = len(values)
        if value_count == 0:
            raise ValueError("staging needs at least one value")
        elif not supports_device_streams(self.device):
            return torch.as_tensor(list(values), dtype=torch.long)
        else:
            pass
        ring = self.staging.rings.get(name)
        if ring is None:
            ring = StagingRing(
                entries=[
                    (
                        PinnedTransferSlot(
                            self.device, torch.long, initial_capacity=self.num_slots + 1
                        ),
                        torch.empty(
                            self.num_slots + 1, dtype=torch.long, device=self.device
                        ),
                    )
                    for _ in range(STAGING_RING_DEPTH)
                ]
            )
            self.staging.rings[name] = ring
        else:
            pass
        ring.turn = (ring.turn + 1) % STAGING_RING_DEPTH
        transfer, device_buffer = ring.entries[ring.turn]
        # note (Zhuofeng-Li): a wrapped host slot may still be supplying its copy.
        if transfer.recorded:
            transfer.synchronize()
        else:
            pass
        host = transfer.view(value_count)
        host.copy_(torch.as_tensor(list(values), dtype=torch.long))
        device_buffer[:value_count].copy_(host, non_blocking=True)
        transfer.record(self.device_module.current_stream(self.device))
        return device_buffer[:value_count]

    def stage_index(self, slots: Sequence[int]) -> torch.Tensor:
        """Stage a cohort's slot ids on the device without a host sync."""
        return self.staged("index", slots)


class CodecStatePool(CodecSlotAllocator):
    """Per-slot codec state where the decode's kernels read and write it: a history pool per
    causal conv, an overlap pool per transposed conv, per-layer K/V rings and frame positions,
    then capture_slots scratch slots past the streams' slots, one per captured row."""

    def __init__(
        self,
        spec: CodecStateSpec,
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
            state_key: torch.zeros(
                total_slots, length, channels, device=self.device, dtype=dtype
            )
            for state_key, channels, length in spec.conv_histories
            if length
        }
        self.overlaps = {
            state_key: torch.zeros(
                total_slots, length, channels, device=self.device, dtype=dtype
            )
            for state_key, channels, length in spec.transconv_overlaps
            if length
        }
        ring_width = spec.num_key_value_heads * spec.head_dim
        self.keys = [
            torch.zeros(
                total_slots * ring_frames, ring_width, device=self.device, dtype=dtype
            )
            for _ in range(spec.num_layers)
        ]
        self.values = [torch.zeros_like(ring) for ring in self.keys]
        self.frame_positions = torch.zeros(
            total_slots, device=self.device, dtype=torch.long
        )
        self.bytes_per_slot = dtype.itemsize * (
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
