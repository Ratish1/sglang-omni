# SPDX-License-Identifier: Apache-2.0
"""Per-stream codec decoder state held by slot: the state's layout, the slot allocator, and the
pool whose histories, overlaps and K/V rings a decode's kernels read and write in place."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sglang_omni.utils.cuda_staging import PinnedTransferSlot
from sglang_omni.utils.device import supports_device_streams


@dataclass(frozen=True)
class CodecStateSpec:
    """Static description of one stream's incremental codec state.

    conv_histories and transconv_overlaps are (key, channels, length) triples in the order the
    decoder visits them. Zero-length entries are kept so every consumer sees the same key set
    regardless of kernel size.
    """

    conv_histories: tuple[tuple[str, int, int], ...]
    transconv_overlaps: tuple[tuple[str, int, int], ...]
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    retained_context: int

    def bytes_per_stream(self, dtype: torch.dtype) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
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
        return int(elements * itemsize)


class CodecSlotAllocator:
    """Bounded, reusable slots for per-stream codec state: a slot is handed out after every
    decode its previous owner queued, and a cohort's slot ids reach the device without a host
    sync."""

    def __init__(self, *, num_slots: int, device: torch.device) -> None:
        if num_slots <= 0:
            raise ValueError("codec state needs at least one slot")
        else:
            pass
        self.device = torch.device(device)
        self.device_module = torch.get_device_module(self.device)
        self._num_slots = int(num_slots)  # noqa: leading-underscore
        self.lock = threading.Lock()
        self.staging = threading.local()
        self.release_events: dict[int, torch.Event] = {}
        self.free: list[int] = list(
            reversed(range(self._num_slots))
        )  # noqa: leading-underscore
        self.retired: set[int] = set()
        self._exhausted_count = 0  # noqa: leading-underscore

    @property
    def num_slots(self) -> int:
        return self._num_slots  # noqa: leading-underscore

    @property
    def exhausted_count(self) -> int:
        return self._exhausted_count  # noqa: leading-underscore

    def active_slots(self) -> int:
        with self.lock:
            return (
                self._num_slots - len(self.free) - len(self.retired)
            )  # noqa: leading-underscore

    def acquire(self) -> int | None:
        """Take a slot once the current stream is ordered after its previous owner's decodes,
        or None when every slot is taken."""
        with self.lock:
            if not self.free:
                self._exhausted_count += 1  # noqa: leading-underscore
                return None
            else:
                pass
            slot = self.free.pop()
            released = self.release_events.pop(slot, None)
        if released is not None:
            self.device_module.current_stream(self.device).wait_event(released)
        else:
            pass
        return slot

    def release(self, slot: int) -> None:
        released = None
        if supports_device_streams(self.device):
            released = self.device_module.Event()
            released.record(self.device_module.current_stream(self.device))
        else:
            pass
        with self.lock:
            if slot in self.retired:
                return
            else:
                pass
            if slot in self.free:
                raise RuntimeError(f"codec state slot {slot} was released twice")
            else:
                pass
            if released is not None:
                self.release_events[slot] = released
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

    STAGING_RING = 4

    def staged(self, name: str, values: Sequence[int]) -> torch.Tensor:
        if not supports_device_streams(self.device):
            return torch.as_tensor(list(values), dtype=torch.long)
        else:
            pass
        count = len(values)
        if count == 0:
            raise ValueError("codec state needs at least one slot")
        else:
            pass
        ring = getattr(self.staging, f"{name}_ring", None)
        if ring is None:
            ring = [
                (
                    PinnedTransferSlot(
                        self.device,
                        torch.long,
                        initial_capacity=self._num_slots
                        + 1,  # noqa: leading-underscore
                    ),
                    torch.empty(
                        self._num_slots + 1,
                        dtype=torch.long,
                        device=self.device,  # noqa: leading-underscore
                    ),
                )
                for _ in range(self.STAGING_RING)
            ]
            setattr(self.staging, f"{name}_ring", ring)
            setattr(self.staging, f"{name}_turn", 0)
        else:
            pass
        turn = (getattr(self.staging, f"{name}_turn") + 1) % self.STAGING_RING
        setattr(self.staging, f"{name}_turn", turn)
        transfer, device = ring[turn]
        # note (Zhuofeng-Li): a wrapped host slot may still be supplying its copy.
        if transfer.recorded:
            transfer.synchronize()
        else:
            pass
        host = transfer.view(count)
        host.copy_(torch.as_tensor(list(values), dtype=torch.long))
        device[:count].copy_(host, non_blocking=True)
        transfer.record(self.device_module.current_stream(self.device))
        return device[:count]

    def stage_index(self, slots: Sequence[int]) -> torch.Tensor:
        """Stage a cohort's slot ids on the device without a host sync."""
        return self.staged("index", slots)


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
