#!/usr/bin/env python3
"""Run an approved repository benchmark after an external readiness check."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

MODULES = {
    "benchmarks.eval.benchmark_omni_mmmu",
    "benchmarks.eval.benchmark_omni_seedtts",
    "benchmarks.eval.benchmark_omni_streaming_ttft",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=sorted(MODULES))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.arguments and args.arguments[0] == "--":
        args.arguments = args.arguments[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    module = importlib.import_module(args.module)
    waiter = getattr(module, "wait_for_service", None)
    entrypoint = getattr(module, "main", None)
    if not callable(waiter) or not callable(entrypoint):
        raise TypeError(f"{args.module} does not expose the expected benchmark API")
    module.wait_for_service = lambda *_args, **_kwargs: None
    sys.argv = [args.module, *args.arguments]
    result = entrypoint()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
