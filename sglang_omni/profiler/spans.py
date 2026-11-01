"""Profiler spans for stage code, on the span primitive of the pinned SGLang.

A span is a torch record_function whenever a torch profiler is active, plus an
NVTX range when SGLANG_ENABLE_NVTX_OPERATIONS=1 and the nvtx package imports.
With neither, every helper returns one shared no-op context.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.nvtx_utils import NVTX_OPERATIONS_ENABLED, operations_nvtx_range
from sglang.srt.utils.profile_utils import build_step_span_name

NULL_SPAN = nullcontext()


def spans_enabled() -> bool:
    return (
        NVTX_OPERATIONS_ENABLED
        or torch.autograd._profiler_enabled()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    )


def span(name: str) -> AbstractContextManager[None]:
    return operations_nvtx_range(name)


def rows_span(name: str, rows: int) -> AbstractContextManager[None]:
    """The span name labeled with the number of rows it serves."""
    if spans_enabled():
        return operations_nvtx_range(f"{name} rows={rows}")
    else:
        return NULL_SPAN


def step_span(
    phase: str, forward_batch: ForwardBatch | None
) -> AbstractContextManager[None]:
    """The phase name labeled with the forward step's mode and batch shape."""
    if not spans_enabled():
        return NULL_SPAN
    elif forward_batch is None:
        return operations_nvtx_range(phase)
    else:
        return operations_nvtx_range(f"{phase} {build_step_span_name(forward_batch)}")
