# SPDX-License-Identifier: Apache-2.0
"""Slot-indexed storage for incremental codec decoder state."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import TypedDict

import torch

from sglang_omni.utils.device import supports_device_streams
from sglang_omni.utils.incremental_codec import (
    IncrementalCodecDecoder,
    IncrementalCodecState,
)


class CodecStateArenaStats(TypedDict):
    slots: int
    active_slots: int
    bytes_per_slot: int
    total_bytes: int
    exhausted: int


class CodecStateArena:
    """Bounded, reusable storage for per-stream incremental codec state.

    A slot's whole state is one row of one buffer, and its frame position one entry of
    another, so gather selects a cohort's rows with one index op and scatter writes the
    advanced state back with one, whatever the number of state tensors. The state tensors
    are views of the row, in the spec's order: conv histories, transposed-conv overlaps,
    then the keys and the values of every layer.

    Note (Qihao Liu): acquire zeroes a slot before handing it out, so a reused slot is
    provably a cold start; release therefore returns a slot without clearing it. A caller
    whose first decode starts from zero_state reads no slot state and skips the zeroing.
    """

    def __init__(
        self,
        decoder: IncrementalCodecDecoder,
        *,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("codec state arena needs at least one slot")
        else:
            pass
        self.device = torch.device(device)
        self.device_module = torch.get_device_module(self.device)
        self.dtype = dtype
        self._num_slots = int(num_slots)  # noqa: leading-underscore
        self.scratch_slot = self._num_slots  # noqa: leading-underscore
        self.spec = decoder.state_spec()
        key_value_shape = (
            self.spec.num_key_value_heads,
            self.spec.retained_context,
            self.spec.head_dim,
        )
        self.state_shapes: tuple[tuple[int, ...], ...] = (
            *((channels, length) for _, channels, length in self.spec.conv_histories),
            *(
                (channels, length)
                for _, channels, length in self.spec.transconv_overlaps
            ),
            *(key_value_shape for _ in range(2 * self.spec.num_layers)),
        )
        self.state_sizes: list[int] = [math.prod(shape) for shape in self.state_shapes]
        self.state_rows = torch.zeros(
            self.num_slots + 1,
            sum(self.state_sizes),
            device=self.device,
            dtype=dtype,
        )
        self.frame_positions = torch.zeros(
            self.num_slots + 1, device=self.device, dtype=torch.long
        )
        self.storage = self.state_of_rows(self.state_rows, self.frame_positions)
        self.lock = threading.Lock()
        self.staging = threading.local()
        self.release_events: dict[int, torch.Event] = {}
        self.free: list[int] = list(
            reversed(range(self._num_slots))
        )  # noqa: leading-underscore
        self.retired: set[int] = set()
        self._exhausted_count = 0  # noqa: leading-underscore
        self._bytes_per_slot = self.spec.bytes_per_stream(
            dtype
        )  # noqa: leading-underscore

    @property
    def num_slots(self) -> int:
        return self._num_slots  # noqa: leading-underscore

    @property
    def bytes_per_slot(self) -> int:
        return self._bytes_per_slot  # noqa: leading-underscore

    @property
    def total_bytes(self) -> int:
        return self._bytes_per_slot * self._num_slots  # noqa: leading-underscore

    @property
    def exhausted_count(self) -> int:
        return self._exhausted_count  # noqa: leading-underscore

    def active_slots(self) -> int:
        with self.lock:
            return (
                self._num_slots - len(self.free) - len(self.retired)
            )  # noqa: leading-underscore

    def acquire(self, *, zeroed: bool = True) -> int | None:
        """Take a slot, zeroed unless the caller's first decode reads none of its state, or
        None when the arena is full."""
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
        if zeroed:
            self.zero_slot(slot)
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

    def zero_slot(self, slot: int) -> None:
        self.state_rows[slot].zero_()
        self.frame_positions[slot].zero_()

    def zero_state(self, count: int) -> IncrementalCodecState:
        """The cold state of count streams, laid out as gathered rows."""
        return self.state_of_rows(
            torch.zeros(
                count, self.state_rows.shape[1], device=self.device, dtype=self.dtype
            ),
            torch.zeros(count, device=self.device, dtype=torch.long),
        )

    def state_of_rows(
        self, rows: torch.Tensor, frame_positions: torch.Tensor
    ) -> IncrementalCodecState:
        """The state whose tensors are views of rows, one row per stream."""
        count = int(rows.shape[0])
        views = (
            piece.view(count, *shape)
            for piece, shape in zip(
                rows.split(self.state_sizes, dim=1), self.state_shapes
            )
        )
        state = IncrementalCodecState(
            transformer_context_length=self.spec.retained_context,
            frame_positions=frame_positions,
        )
        for key, _, _ in self.spec.conv_histories:
            state.conv_histories[key] = next(views)
        for key, _, _ in self.spec.transconv_overlaps:
            state.transconv_overlaps[key] = next(views)
        for layer_index in range(self.spec.num_layers):
            state.transformer_keys[layer_index] = next(views)
        for layer_index in range(self.spec.num_layers):
            state.transformer_values[layer_index] = next(views)
        return state

    def rows_of_state(self, state: IncrementalCodecState, count: int) -> torch.Tensor:
        """The state's tensors packed into one row per stream, in the order of the views."""
        pieces = [
            *(state.conv_histories[key] for key, _, _ in self.spec.conv_histories),
            *(
                state.transconv_overlaps[key]
                for key, _, _ in self.spec.transconv_overlaps
            ),
            *(
                state.transformer_keys[layer_index]
                for layer_index in range(self.spec.num_layers)
            ),
            *(
                state.transformer_values[layer_index]
                for layer_index in range(self.spec.num_layers)
            ),
        ]
        return torch.cat([piece.reshape(count, -1) for piece in pieces], dim=1)

    STAGING_RING = 4

    def staged(self, name: str, values: Sequence[int]) -> torch.Tensor:
        if not supports_device_streams(self.device):
            return torch.as_tensor(list(values), dtype=torch.long)
        else:
            pass
        count = len(values)
        if count == 0:
            raise ValueError("codec state arena needs at least one slot")
        else:
            pass
        ring = getattr(self.staging, f"{name}_ring", None)
        if ring is None:
            ring = [
                (
                    torch.empty(
                        self._num_slots + 1, dtype=torch.long
                    ).pin_memory(),  # noqa: leading-underscore
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
        host, device = ring[turn]
        host[:count].copy_(torch.as_tensor(list(values), dtype=torch.long))
        device[:count].copy_(host[:count], non_blocking=True)
        return device[:count]

    def stage_index(self, slots: Sequence[int]) -> torch.Tensor:
        """Stage a cohort's slot ids on the device without a host sync."""
        return self.staged("index", slots)

    def gather(self, slots: Sequence[int]) -> IncrementalCodecState:
        """Select a cohort's rows into one contiguous state."""
        return self.gather_by_index(self.stage_index(slots))

    def gather_by_index(self, index: torch.Tensor) -> IncrementalCodecState:
        """Select the rows named by a device index tensor.

        Capturable: with a static index this is the gather half of a graph
        that reads the arena directly.
        """
        return self.state_of_rows(
            self.state_rows.index_select(0, index),
            self.frame_positions.index_select(0, index),
        )

    def scatter(self, slots: Sequence[int], state: IncrementalCodecState) -> None:
        """Write an advanced cohort state back into its slots."""
        self.scatter_by_index(self.stage_index(slots), state)

    def scatter_by_index(
        self, index: torch.Tensor, state: IncrementalCodecState
    ) -> None:
        """Write a cohort state into the rows named by a device index tensor."""
        count = int(index.shape[0])
        frame_positions = state.frame_positions
        if frame_positions is None:
            raise RuntimeError("codec state arena requires per-row frame positions")
        elif tuple(frame_positions.shape) != (count,):
            raise RuntimeError(
                f"codec state arena expected {(count,)} for frame_positions, "
                f"got {tuple(frame_positions.shape)}"
            )
        elif frame_positions.dtype != torch.long:
            raise RuntimeError(
                f"codec state arena expected {torch.long} for frame_positions, "
                f"got {frame_positions.dtype}"
            )
        else:
            pass
        self.state_rows.index_copy_(0, index, self.rows_of_state(state, count))
        self.frame_positions.index_copy_(0, index, frame_positions)

    def describe(self) -> CodecStateArenaStats:
        return {
            "slots": self._num_slots,  # noqa: leading-underscore
            "active_slots": self.active_slots(),
            "bytes_per_slot": self._bytes_per_slot,  # noqa: leading-underscore
            "total_bytes": self.total_bytes,
            "exhausted": self._exhausted_count,  # noqa: leading-underscore
        }


__all__ = ["CodecStateArena", "CodecStateArenaStats"]
