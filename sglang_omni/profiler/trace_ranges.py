# SPDX-License-Identifier: Apache-2.0
"""Small profiling range helpers.

PyTorch ``record_function`` is the primary trace surface. Optional NVTX ranges
are useful when Chrome traces drop user labels or when inspecting with
Nsight/Perfetto. NVTX is env-gated to keep normal serving overhead unchanged.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch

_NVTX_ENABLED = os.environ.get("SGLANG_OMNI_NVTX_RANGES") == "1"


@contextmanager
def profile_range(name: str) -> Iterator[None]:
    """Enter a PyTorch profiler range and, optionally, an NVTX range."""
    nvtx_ctx = nullcontext()
    if _NVTX_ENABLED and torch.cuda.is_available():
        nvtx_ctx = torch.cuda.nvtx.range(name)
    with torch.profiler.record_function(name), nvtx_ctx:
        yield
