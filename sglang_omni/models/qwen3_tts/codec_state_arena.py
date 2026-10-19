"""Slot-indexed storage for Qwen3-TTS incremental Codec decoder state."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import torch

from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)
from sglang_omni.utils.cuda_staging import PinnedTransferSlot
from sglang_omni.utils.device import supports_device_streams

if TYPE_CHECKING:
    from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
        IncrementalCodecGraphStats,
    )
else:
    pass


class DisabledCodecGraphStats(TypedDict):
    enabled: bool


class CodecCudaGraphStats(TypedDict):
    cold: IncrementalCodecGraphStats | DisabledCodecGraphStats
    window: IncrementalCodecGraphStats | DisabledCodecGraphStats
    tail: IncrementalCodecGraphStats | DisabledCodecGraphStats
    warm: list[IncrementalCodecGraphStats]


class ReferenceStateStats(TypedDict):
    """Reference states and the arena slots they hold, which streams take back on demand."""

    references: int
    slots: int
    hits: int
    fills: int
    evictions: int


class CodecStateStats(TypedDict, total=False):
    """Arena/scheduler snapshot; a disabled decoder reports only enabled."""

    slots: int
    active_slots: int
    bytes_per_slot: int
    total_bytes: int
    exhausted: int
    enabled: bool
    left_context_fallbacks: int
    cuda_graphs: CodecCudaGraphStats
    reference_states: ReferenceStateStats


class Qwen3TTSCodecStateArena:
    """Bounded, reusable storage for per-stream incremental Codec state.

    Every buffer from ``Qwen3TTSIncrementalDecoder.state_spec`` carries a
    leading slot dimension, so the arena is exactly the state a decode of
    ``num_slots`` rows would use: ``gather`` selects a cohort's rows into a
    contiguous state and ``scatter`` writes the advanced state back.

    Note (Qihao Liu): ``acquire`` zeroes a slot before handing it out, so a
    reused slot is provably a cold start; ``release`` therefore returns a slot
    without clearing it.
    """

    def __init__(
        self,
        decoder: Qwen3TTSIncrementalDecoder,
        *,
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("Qwen3-TTS codec state arena needs at least one slot")
        else:
            pass
        self.decoder = decoder
        self.device = torch.device(device)
        self.device_module = torch.get_device_module(self.device)
        self.dtype = dtype
        self._num_slots = int(num_slots)  # noqa: leading-underscore
        self.scratch_slot = self._num_slots  # noqa: leading-underscore
        self.storage = decoder.init_state(
            self._num_slots + 1,
            device=self.device,
            dtype=dtype,  # noqa: leading-underscore
        )
        self.lock = threading.Lock()
        self.staging = threading.local()
        self.release_events: dict[int, torch.Event] = {}
        self.free: list[int] = list(
            reversed(range(self._num_slots))
        )  # noqa: leading-underscore
        self.retired: set[int] = set()
        self._exhausted_count = 0  # noqa: leading-underscore
        spec = decoder.state_spec()
        self._bytes_per_slot = spec.bytes_per_stream(dtype)  # noqa: leading-underscore

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

    def has_free_slot(self) -> bool:
        with self.lock:
            return bool(self.free)

    def acquire(self, *, source_slot: int | None = None) -> int | None:
        """Take a slot, zeroed or holding a copy of source_slot's state; None when the arena is full."""
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
        if source_slot is None:
            self.zero_slot(slot)
        else:
            self.copy_slot(source_slot, slot)
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
                raise RuntimeError(
                    f"Qwen3-TTS codec state slot {slot} was released twice"
                )
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

    def _buffers(self, state: Qwen3TTSIncrementalCodecState) -> list[torch.Tensor]:
        return [
            *state.conv_histories.values(),
            *state.transconv_overlaps.values(),
            *state.transformer_keys.values(),
            *state.transformer_values.values(),
        ]

    def zero_slot(self, slot: int) -> None:
        # note (ratish): one multi-tensor launch for the slot's buffers: an initial decode resets
        # a fresh slot per row, and every row of its batch waits for all the resets before launch.
        torch._foreach_zero_(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            [buffer[slot] for buffer in self._buffers(self.storage)]
        )
        self.storage.frame_positions[slot].zero_()

    def copy_slot(self, source_slot: int, slot: int) -> None:
        buffers = self._buffers(self.storage)
        torch._foreach_copy_(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            [buffer[slot] for buffer in buffers],
            [buffer[source_slot] for buffer in buffers],
        )
        self.storage.frame_positions[slot].copy_(
            self.storage.frame_positions[source_slot]
        )

    STAGING_RING = 4

    def staged(self, name: str, values: Sequence[int]) -> torch.Tensor:
        if not supports_device_streams(self.device):
            return torch.as_tensor(list(values), dtype=torch.long)
        else:
            pass
        count = len(values)
        if count == 0:
            raise ValueError("Qwen3-TTS codec state arena needs at least one slot")
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

    def gather(self, slots: Sequence[int]) -> Qwen3TTSIncrementalCodecState:
        """Select a cohort's rows into one contiguous state."""
        return self.gather_by_index(self.stage_index(slots))

    def gather_by_index(self, index: torch.Tensor) -> Qwen3TTSIncrementalCodecState:
        """Select the rows named by a device index tensor.

        Capturable: with a static index this is the gather half of a graph
        that reads the arena directly.
        """
        storage = self.storage
        state = Qwen3TTSIncrementalCodecState(
            transformer_context_length=storage.transformer_context_length,
            frame_positions=storage.frame_positions.index_select(0, index),
        )
        for key, buffer in storage.conv_histories.items():
            state.conv_histories[key] = buffer.index_select(0, index)
        for key, buffer in storage.transconv_overlaps.items():
            state.transconv_overlaps[key] = buffer.index_select(0, index)
        for layer_index, buffer in storage.transformer_keys.items():
            state.transformer_keys[layer_index] = buffer.index_select(0, index)
        for layer_index, buffer in storage.transformer_values.items():
            state.transformer_values[layer_index] = buffer.index_select(0, index)
        return state

    def scatter(
        self, slots: Sequence[int], state: Qwen3TTSIncrementalCodecState
    ) -> None:
        """Write an advanced cohort state back into its slots."""
        self.scatter_by_index(self.stage_index(slots), state)

    def scatter_by_index(
        self, index: torch.Tensor, state: Qwen3TTSIncrementalCodecState
    ) -> None:
        """Write a cohort state into the rows named by a device index tensor."""
        storage = self.storage
        if state.frame_positions is None:
            raise RuntimeError(
                "Qwen3-TTS codec state arena requires per-row frame positions"
            )
        else:
            pass
        self.copy_rows(storage.frame_positions, index, state.frame_positions)
        for key, buffer in storage.conv_histories.items():
            self.copy_rows(buffer, index, state.conv_histories[key], key)
        for key, buffer in storage.transconv_overlaps.items():
            self.copy_rows(buffer, index, state.transconv_overlaps[key], key)
        for layer_index, buffer in storage.transformer_keys.items():
            self.copy_rows(
                buffer, index, state.transformer_keys[layer_index], f"key.{layer_index}"
            )
        for layer_index, buffer in storage.transformer_values.items():
            self.copy_rows(
                buffer,
                index,
                state.transformer_values[layer_index],
                f"value.{layer_index}",
            )

    @staticmethod
    def copy_rows(
        buffer: torch.Tensor,
        index: torch.Tensor,
        rows: torch.Tensor,
        key: str = "frame_positions",
    ) -> None:
        expected = (int(index.shape[0]), *buffer.shape[1:])
        if tuple(rows.shape) != expected:
            raise RuntimeError(
                f"Qwen3-TTS codec state arena expected {expected} for {key}, got {tuple(rows.shape)}"
            )
        else:
            pass
        if rows.dtype != buffer.dtype:
            raise RuntimeError(
                f"Qwen3-TTS codec state arena expected {buffer.dtype} for {key}, got {rows.dtype}"
            )
        else:
            pass
        buffer.index_copy_(0, index, rows.contiguous())

    def describe(self) -> CodecStateStats:
        return {
            "slots": self._num_slots,  # noqa: leading-underscore
            "active_slots": self.active_slots(),
            "bytes_per_slot": self._bytes_per_slot,  # noqa: leading-underscore
            "total_bytes": self.total_bytes,
            "exhausted": self._exhausted_count,  # noqa: leading-underscore
        }


# note (ratish): a reference's state is filled at its second use, so a voice used
# once never adds a fill row to its first chunk's decode.
REFERENCE_STATE_FILL_USES = 2


@dataclass(eq=False)
class ReferenceState:
    """One reference's codec state; slot is None until its fill is admitted."""

    uses: int = 0
    slot: int | None = None
    is_ready: bool = False
    lock_count: int = 0


class Qwen3TTSReferenceStateCache:
    """Codec state after a voice's reference codes, keyed by the codes' SHA-256.

    Entries hold arena slots and give them back, least recent first, when a stream
    needs one. Only the initial worker thread calls it, on its decode stream.
    """

    def __init__(self, arena: Qwen3TTSCodecStateArena) -> None:
        self.arena = arena
        self.states: OrderedDict[str, ReferenceState] = OrderedDict()
        self.hits = 0
        self.fills = 0
        self.evictions = 0

    def use(self, reference_key: str) -> ReferenceState:
        """Count one first chunk of the reference and make it the most recent."""
        reference_state = self.states.get(reference_key)
        if reference_state is None:
            if len(self.states) >= self.arena.num_slots:
                self.evict_least_recent(needs_slot=False)
            else:
                pass
            reference_state = ReferenceState()
            self.states[reference_key] = reference_state
        else:
            self.states.move_to_end(reference_key)
        reference_state.uses += 1
        return reference_state

    def acquire_stream_slot(self, reference_state: ReferenceState | None) -> int | None:
        """A stream's slot: a copy of the reference's state once it is ready, zeroed otherwise."""
        if reference_state is not None and reference_state.is_ready:
            slot = self.acquire_slot(reference_state, source_slot=reference_state.slot)
            if slot is not None:
                self.hits += 1
            else:
                pass
            return slot
        else:
            return self.acquire_slot(reference_state, source_slot=None)

    def start_fill(self, reference_state: ReferenceState) -> int | None:
        """A zeroed slot for the reference's state, locked until finish_fill."""
        if (
            reference_state.uses < REFERENCE_STATE_FILL_USES
            or reference_state.slot is not None
        ):
            return None
        else:
            pass
        slot = self.acquire_slot(reference_state, source_slot=None)
        if slot is None:
            return None
        else:
            pass
        reference_state.slot = slot
        reference_state.lock_count += 1
        self.fills += 1
        return slot

    def finish_fill(self, reference_state: ReferenceState, *, is_filled: bool) -> None:
        slot = reference_state.slot
        assert slot is not None
        reference_state.lock_count -= 1
        if is_filled:
            reference_state.is_ready = True
        else:
            self.arena.release(slot)
            reference_state.slot = None

    def acquire_slot(
        self, reference_state: ReferenceState | None, *, source_slot: int | None
    ) -> int | None:
        """Acquire from the arena, evicting least recent entries other than reference_state first."""
        if reference_state is not None:
            reference_state.lock_count += 1
        else:
            pass
        while not self.arena.has_free_slot() and self.evict_least_recent(
            needs_slot=True
        ):
            pass
        if reference_state is not None:
            reference_state.lock_count -= 1
        else:
            pass
        return self.arena.acquire(source_slot=source_slot)

    def evict_least_recent(self, *, needs_slot: bool) -> bool:
        """Drop the least recent unlocked entry, one holding a slot when needs_slot."""
        for reference_key, reference_state in self.states.items():
            if reference_state.lock_count == 0 and (
                reference_state.slot is not None or not needs_slot
            ):
                self.states.pop(reference_key)
                if reference_state.slot is not None:
                    self.arena.release(reference_state.slot)
                    self.evictions += 1
                else:
                    pass
                return True
            else:
                pass
        return False

    def describe(self) -> ReferenceStateStats:
        return {
            "references": len(self.states),
            "slots": sum(
                reference_state.slot is not None
                for reference_state in self.states.values()
            ),
            "hits": self.hits,
            "fills": self.fills,
            "evictions": self.evictions,
        }


__all__ = ["Qwen3TTSCodecStateArena", "Qwen3TTSReferenceStateCache"]
