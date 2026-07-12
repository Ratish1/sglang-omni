# SPDX-License-Identifier: Apache-2.0
"""Lightweight scheduler message types shared across scheduling backends."""

from __future__ import annotations

import queue
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

import torch


@dataclass(frozen=True)
class CudaReadyEvent:
    """Producer-stream completion carried across the scheduler outbox."""

    device: int
    event: torch.cuda.Event


def _cuda_devices(obj: Any, devices: set[int], seen: set[int]) -> None:
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            device = obj.device.index
            if device is None:
                raise RuntimeError("CUDA tensor has no device index")
            devices.add(device)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _cuda_devices(value, devices, seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            _cuda_devices(value, devices, seen)
        return
    if is_dataclass(obj) and not isinstance(obj, type):
        for item in fields(obj):
            _cuda_devices(getattr(obj, item.name), devices, seen)


@dataclass
class IncomingMessage:
    request_id: str
    type: Literal["new_request", "stream_chunk", "stream_done"]
    data: Any = None


@dataclass
class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error"]
    data: Any = None
    target: str | None = None
    metadata: dict[str, Any] | None = None
    _cuda_ready: tuple[CudaReadyEvent, ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def record_cuda_ready(self) -> None:
        """Record completion after the producer's work on each CUDA device.

        The scheduler must publish the message while its current CUDA stream is
        ordered after every write to the message's tensors. Side streams must
        therefore be joined before publishing.
        """
        devices: set[int] = set()
        _cuda_devices(self.data, devices, set())
        _cuda_devices(self.metadata, devices, set())
        ready = []
        for device in sorted(devices):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            ready.append(CudaReadyEvent(device=device, event=event))
        self._cuda_ready = tuple(ready)

    def wait_cuda_ready(self) -> None:
        """Order the current CUDA streams after the scheduler producer."""
        for ready in self._cuda_ready:
            torch.cuda.current_stream(ready.device).wait_event(ready.event)


class SchedulerOutputQueue(queue.Queue[OutgoingMessage]):
    """Scheduler outbox whose put operation transfers CUDA readiness."""

    def put(
        self,
        item: OutgoingMessage,
        block: bool = True,
        timeout: float | None = None,
    ) -> None:
        item.record_cuda_ready()
        super().put(item, block=block, timeout=timeout)
