# SPDX-License-Identifier: Apache-2.0
"""Startup-only sampling switches and opt-in Higgs trace ranges."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import torch

GUMBEL_ENV = "SGLANG_OMNI_HIGGS_USE_GUMBEL_SAMPLE"
PROFILE_ENV = "SGLANG_OMNI_HIGGS_PROFILE_SAMPLING"


def _read_switch(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


# A captured graph keeps the branch selected at capture. Restart to change it.
USE_GUMBEL_SAMPLE = _read_switch(GUMBEL_ENV)
PROFILE_SAMPLING = _read_switch(PROFILE_ENV)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def profile_sampling(
    name: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Return the original callable when tracing is disabled.

    Python ranges inside a CUDA graph describe capture, not each replay.
    Use the runner's forward range and CUDA node tracing for replay attribution.
    """

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        if not PROFILE_SAMPLING:
            return function

        @wraps(function)
        def traced(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with torch.profiler.record_function(name):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(name)
                    try:
                        return function(*args, **kwargs)
                    finally:
                        torch.cuda.nvtx.range_pop()
                return function(*args, **kwargs)

        return traced

    return decorate
