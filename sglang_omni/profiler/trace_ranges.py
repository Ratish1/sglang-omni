# SPDX-License-Identifier: Apache-2.0
"""Small profiling range helpers.

PyTorch ``record_function`` is useful while the torch profiler is active, but
entering it in the serving hot path is not free. Keep normal serving on a
``nullcontext`` and only emit ranges when a profiler is running or NVTX is
explicitly requested.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import ContextManager, Iterator

import torch

from sglang_omni.profiler.torch_profiler import TorchProfiler

_NVTX_ENABLED = os.environ.get("SGLANG_OMNI_NVTX_RANGES") == "1"
_NULL_RANGE = nullcontext()


@contextmanager
def _combined_range(name: str) -> Iterator[None]:
    with torch.profiler.record_function(name), torch.cuda.nvtx.range(name):
        yield


def profile_range(name: str) -> ContextManager[None]:
    """Return a profiler range only when profiling is enabled."""
    torch_active = TorchProfiler.is_active()
    nvtx_active = _NVTX_ENABLED and torch.cuda.is_available()
    if not torch_active and not nvtx_active:
        return _NULL_RANGE
    if torch_active and nvtx_active:
        return _combined_range(name)
    if torch_active:
        return torch.profiler.record_function(name)
    return torch.cuda.nvtx.range(name)
