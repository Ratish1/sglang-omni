# SPDX-License-Identifier: Apache-2.0
"""Opt-in host annotations for process-tree Nsight Systems captures.

Set SGLANG_OMNI_PIPELINE_NVTX=1 before starting the server. No CUDA events,
synchronization, tensor reads, or profiler lifecycle are introduced here.
Ranges describe host submission; CUDA correlation identifies device execution.
"""

from __future__ import annotations

import functools
import json
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, ParamSpec, TypeVar

ENABLED = os.environ.get("SGLANG_OMNI_PIPELINE_NVTX", "0") == "1"
PREFIX = "omni.pipeline:"
_P = ParamSpec("_P")
_R = TypeVar("_R")

if ENABLED:
    import nvtx


def _summary(value: Any) -> Any:
    # Even a scalar CUDA tensor must never be read by diagnostic metadata.
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(getattr(value, "device", "unknown")),
        }
    return {"type": type(value).__name__}


def _message(stage: str, op: str, metadata: dict[str, Any]) -> str:
    return PREFIX + json.dumps(
        {"stage": stage, "op": op, **metadata},
        separators=(",", ":"),
        default=_summary,
    )


def trace_range(stage: str, op: str, **metadata: Any) -> AbstractContextManager:
    """A synchronous host range. Never hold this context across an await."""
    if not ENABLED:
        return nullcontext()
    return nvtx.annotate(
        _message(stage, op, {**metadata, "kind": "range"}), domain="sglang_omni"
    )


def trace_call(
    stage: str,
    op: str,
    metadata: Callable[_P, dict[str, Any]] | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Annotate a synchronous function; the disabled decorator is identity."""

    def decorate(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        if not ENABLED:
            return fn

        @functools.wraps(fn)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            fields = metadata(*args, **kwargs) if metadata else {}
            with trace_range(stage, op, **fields):
                return fn(*args, **kwargs)

        return wrapped

    return decorate


def mark(stage: str, op: str, **metadata: Any) -> None:
    if ENABLED:
        nvtx.mark(
            _message(stage, op, {**metadata, "kind": "mark"}), domain="sglang_omni"
        )
