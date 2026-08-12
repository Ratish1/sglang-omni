# SPDX-License-Identifier: Apache-2.0
"""Calibrate ``torch.cuda.set_sync_debug_mode`` in the serving environment.

Run exactly one case per Python process.  The detector is experimental and its
coverage varies by Torch build, so this probe records observed warnings/errors
before a clean server log is interpreted as evidence.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable
from typing import Any

import torch

_CASE_NAMES = (
    "item",
    "dtoh_pageable",
    "h2d_pageable",
    "h2d_pinned_nonblocking",
    "dtoh_pinned_nonblocking",
    "stream_synchronize",
    "device_synchronize",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=_CASE_NAMES)
    parser.add_argument("mode", choices=("warn", "error"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--elements", type=int, default=1_048_576)
    return parser


def run_probe(
    case: str,
    mode: str,
    *,
    device_index: int,
    elements: int,
) -> dict[str, Any]:
    if elements <= 0:
        raise ValueError("elements must be > 0")
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    # All allocation and setup happens before the detector is armed so each
    # result describes only the named operation.
    gpu = torch.ones(elements, device=device)
    scalar = gpu[0]
    pageable = torch.ones_like(gpu, device="cpu")
    pinned = torch.empty_like(gpu, device="cpu", pin_memory=True)
    work_stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)

    def stream_synchronize() -> None:
        with torch.cuda.stream(work_stream):
            torch.sin(gpu)
        work_stream.synchronize()

    operations: dict[str, Callable[[], object]] = {
        "item": scalar.item,
        "dtoh_pageable": gpu.cpu,
        "h2d_pageable": lambda: pageable.to(device),
        "h2d_pinned_nonblocking": lambda: pinned.to(device, non_blocking=True),
        "dtoh_pinned_nonblocking": lambda: pinned.copy_(gpu, non_blocking=True),
        "stream_synchronize": stream_synchronize,
        "device_synchronize": lambda: torch.cuda.synchronize(device),
    }

    result: dict[str, Any] = {
        "case": case,
        "mode": mode,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "device_capability": list(torch.cuda.get_device_capability(device_index)),
        "outcome": "return",
        "exception": None,
        "warnings": [],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode(mode)
        try:
            operations[case]()
        except Exception as exc:  # The error mode is expected to raise.
            result["outcome"] = "raise"
            result["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        finally:
            torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize(device)
        result["warnings"] = [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "filename": warning.filename,
                "lineno": warning.lineno,
            }
            for warning in caught
        ]
    return result


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_probe(
        args.case,
        args.mode,
        device_index=args.device,
        elements=args.elements,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
