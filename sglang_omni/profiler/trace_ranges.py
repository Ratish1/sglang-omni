# SPDX-License-Identifier: Apache-2.0
"""Opt-in semantic ranges shared by JSONL, PyTorch, and Nsight captures."""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch.profiler import record_function

from sglang_omni.profiler.event_recorder import emit, get_recorder
from sglang_omni.profiler.torch_profiler import TorchProfiler

_NVTX_ENABLED = os.environ.get("SGLANG_OMNI_NVTX_RANGES", "").strip() == "1"
_NOOP = contextlib.nullcontext()
_NVTX_WINDOW_RUN_ID: str | None = None
_NVTX_WINDOW_OWNER: int | None = None
_NVTX_WINDOW_NATIVE_TID: int | None = None


@contextlib.contextmanager
def _combined_range(name: str) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        if TorchProfiler.is_active() and TorchProfiler.is_owner_thread():
            stack.enter_context(record_function(name))
        if _NVTX_ENABLED and torch.cuda.is_available():
            stack.enter_context(torch.cuda.nvtx.range(name))
        yield


def trace_range(name: str):
    """Return a no-op unless torch-owner or explicit NVTX capture is active."""
    if (
        not (TorchProfiler.is_active() and TorchProfiler.is_owner_thread())
        and not _NVTX_ENABLED
    ):
        return _NOOP
    return _combined_range(name)


def start_async_trace_range(name: str) -> int | None:
    """Start an NVTX range whose end is not lexically scoped.

    Async ranges are only for Nsight captures.  Unlike ``range_push`` /
    ``range_pop``, NVTX start/end ranges may span scheduler iterations or
    threads.  Returning ``None`` keeps the disabled path allocation-free and
    gives callers an explicit token to pass to ``end_async_trace_range``.
    """

    if not _NVTX_ENABLED or not torch.cuda.is_available():
        return None
    return int(torch.cuda.nvtx.range_start(name))


def end_async_trace_range(range_id: int | None) -> None:
    """End a range returned by ``start_async_trace_range``."""

    if range_id is not None:
        torch.cuda.nvtx.range_end(range_id)


@contextlib.contextmanager
def profile_span(
    *,
    request_id: str,
    stage: str | None,
    name: str,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Emit balanced semantic events and an optional framework/system range."""

    events_active = get_recorder().is_active()
    ranges_active = (
        TorchProfiler.is_active() and TorchProfiler.is_owner_thread()
    ) or _NVTX_ENABLED
    if not events_active and not ranges_active:
        yield
        return

    start_ns = time.monotonic_ns()
    if events_active:
        emit(
            request_id=request_id,
            stage=stage,
            event_name=f"{name}_start",
            metadata=metadata,
        )
    status = "ok"
    try:
        with trace_range(name):
            yield
    except BaseException:
        status = "error"
        raise
    finally:
        if events_active:
            end_metadata = dict(metadata or {})
            end_metadata.update(
                {
                    "duration_ns": time.monotonic_ns() - start_ns,
                    "status": status,
                }
            )
            emit(
                request_id=request_id,
                stage=stage,
                event_name=f"{name}_end",
                metadata=end_metadata,
            )


def nvtx_enabled() -> bool:
    return _NVTX_ENABLED


def start_nvtx_window(run_id: str | None) -> dict[str, Any]:
    """Push the Nsight capture window on its scheduler owner thread."""
    global _NVTX_WINDOW_NATIVE_TID, _NVTX_WINDOW_OWNER, _NVTX_WINDOW_RUN_ID
    if not _NVTX_ENABLED:
        raise RuntimeError(
            "SGLANG_OMNI_NVTX_RANGES=1 is required for an NVTX capture window"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the NVTX capture window")
    owner = threading.get_ident()
    if _NVTX_WINDOW_OWNER is not None:
        if _NVTX_WINDOW_OWNER == owner and _NVTX_WINDOW_RUN_ID == run_id:
            return nvtx_window_snapshot()
        raise RuntimeError(
            f"NVTX window already active for run_id={_NVTX_WINDOW_RUN_ID}"
        )
    torch.cuda.nvtx.range_push("sglang_omni.capture_window")
    _NVTX_WINDOW_OWNER = owner
    _NVTX_WINDOW_NATIVE_TID = threading.get_native_id()
    _NVTX_WINDOW_RUN_ID = run_id
    return nvtx_window_snapshot()


def stop_nvtx_window(run_id: str | None) -> dict[str, Any] | None:
    """Pop the Nsight capture window on the thread that pushed it."""
    global _NVTX_WINDOW_NATIVE_TID, _NVTX_WINDOW_OWNER, _NVTX_WINDOW_RUN_ID
    if _NVTX_WINDOW_OWNER is None:
        return None
    if _NVTX_WINDOW_OWNER != threading.get_ident():
        raise RuntimeError("NVTX capture window must stop on its owner thread")
    if (
        run_id is not None
        and _NVTX_WINDOW_RUN_ID is not None
        and run_id != _NVTX_WINDOW_RUN_ID
    ):
        raise RuntimeError(
            f"NVTX stop run_id={run_id} does not match "
            f"active run_id={_NVTX_WINDOW_RUN_ID}"
        )
    snapshot = nvtx_window_snapshot()
    torch.cuda.nvtx.range_pop()
    _NVTX_WINDOW_OWNER = None
    _NVTX_WINDOW_NATIVE_TID = None
    _NVTX_WINDOW_RUN_ID = None
    snapshot["active"] = False
    return snapshot


def nvtx_window_snapshot() -> dict[str, Any]:
    return {
        "active": _NVTX_WINDOW_OWNER is not None,
        "run_id": _NVTX_WINDOW_RUN_ID,
        "owner_tid": _NVTX_WINDOW_NATIVE_TID,
        "owner_ident": _NVTX_WINDOW_OWNER,
        "owner_thread": (
            threading.current_thread().name
            if _NVTX_WINDOW_OWNER == threading.get_ident()
            else None
        ),
    }


__all__ = [
    "end_async_trace_range",
    "nvtx_enabled",
    "nvtx_window_snapshot",
    "profile_span",
    "start_async_trace_range",
    "start_nvtx_window",
    "stop_nvtx_window",
    "trace_range",
]
