# SPDX-License-Identifier: Apache-2.0
"""Small metadata helpers for request-level profiling events.

The event recorder already protects JSON serialization from accidentally
materializing tensors. These helpers keep instrumentation callsites consistent:
count tensors recursively, report byte sizes, and time small code regions with
monotonic clocks.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class TensorStats:
    tensor_count: int = 0
    tensor_bytes: int = 0
    cuda_tensor_count: int = 0
    cpu_tensor_count: int = 0
    other_device_tensor_count: int = 0
    devices: tuple[str, ...] = ()
    dtypes: tuple[str, ...] = ()

    def to_metadata(self, *, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}tensor_count": self.tensor_count,
            f"{prefix}tensor_bytes": self.tensor_bytes,
            f"{prefix}cuda_tensor_count": self.cuda_tensor_count,
            f"{prefix}cpu_tensor_count": self.cpu_tensor_count,
            f"{prefix}other_device_tensor_count": self.other_device_tensor_count,
            f"{prefix}tensor_devices": list(self.devices),
            f"{prefix}tensor_dtypes": list(self.dtypes),
        }


@dataclass
class ElapsedTimer:
    start_ns: int
    end_ns: int | None = None

    @classmethod
    def start(cls) -> "ElapsedTimer":
        return cls(start_ns=time.perf_counter_ns())

    def stop(self) -> int:
        self.end_ns = time.perf_counter_ns()
        return self.elapsed_ns

    @property
    def elapsed_ns(self) -> int:
        end_ns = self.end_ns if self.end_ns is not None else time.perf_counter_ns()
        return max(0, end_ns - self.start_ns)

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1e6


@contextmanager
def elapsed_timer() -> Iterator[ElapsedTimer]:
    timer = ElapsedTimer.start()
    try:
        yield timer
    finally:
        timer.stop()


def tensor_stats(value: Any) -> TensorStats:
    seen: set[int] = set()
    devices: set[str] = set()
    dtypes: set[str] = set()
    counts = {
        "tensor_count": 0,
        "tensor_bytes": 0,
        "cuda_tensor_count": 0,
        "cpu_tensor_count": 0,
        "other_device_tensor_count": 0,
    }

    def visit(obj: Any) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if _is_tensor_like(obj):
            counts["tensor_count"] += 1
            counts["tensor_bytes"] += _tensor_nbytes(obj)
            device = str(getattr(obj, "device", "unknown"))
            dtype = str(getattr(obj, "dtype", "unknown"))
            devices.add(device)
            dtypes.add(dtype)
            if device.startswith("cuda"):
                counts["cuda_tensor_count"] += 1
            elif device == "cpu":
                counts["cpu_tensor_count"] += 1
            else:
                counts["other_device_tensor_count"] += 1
            return

        if isinstance(obj, dict):
            for key, item in obj.items():
                visit(key)
                visit(item)
            return
        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                visit(item)

    visit(value)
    return TensorStats(
        tensor_count=counts["tensor_count"],
        tensor_bytes=counts["tensor_bytes"],
        cuda_tensor_count=counts["cuda_tensor_count"],
        cpu_tensor_count=counts["cpu_tensor_count"],
        other_device_tensor_count=counts["other_device_tensor_count"],
        devices=tuple(sorted(devices)),
        dtypes=tuple(sorted(dtypes)),
    )


def object_nbytes(value: Any) -> int | None:
    if value is None:
        return 0
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        try:
            return int(nbytes)
        except (TypeError, ValueError):
            return None
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel() * element_size())
        except (TypeError, ValueError):
            return None
    return None


def object_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def object_summary(value: Any, *, prefix: str = "") -> dict[str, Any]:
    stats = tensor_stats(value)
    metadata = stats.to_metadata(prefix=prefix)
    nbytes = object_nbytes(value)
    shape = object_shape(value)
    if nbytes is not None:
        metadata[f"{prefix}bytes"] = nbytes
    if shape is not None:
        metadata[f"{prefix}shape"] = shape
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        metadata[f"{prefix}dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        metadata[f"{prefix}device"] = str(device)
    return metadata


def _is_tensor_like(value: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and (hasattr(value, "numel") or hasattr(value, "nbytes"))
    )


def _tensor_nbytes(value: Any) -> int:
    nbytes = object_nbytes(value)
    return int(nbytes) if nbytes is not None else 0
