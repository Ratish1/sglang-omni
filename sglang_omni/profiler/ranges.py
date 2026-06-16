# SPDX-License-Identifier: Apache-2.0
"""Small helpers for optional torch profiler ranges."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from sglang_omni.profiler.torch_profiler import TorchProfiler


def torch_profile_range(name: str):
    if not TorchProfiler.is_active():
        return nullcontext()
    return torch.profiler.record_function(name)
