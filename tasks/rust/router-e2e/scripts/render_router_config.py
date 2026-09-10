#!/usr/bin/env python3
"""Render the current CI Rust-router configuration for an external worker pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from tests.test_model.rust_router_config import (  # noqa: E402
    CiRouterTopology,
    render_router_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=tuple(CiRouterTopology), required=True)
    parser.add_argument(
        "--policy", choices=("round_robin", "least_requests"), required=True
    )
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--worker-url", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.worker_url) != 2:
        parser.error("exactly two --worker-url values are required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    text = render_router_config(
        topology=CiRouterTopology(args.topology),
        router_port=args.router_port,
        worker_urls=args.worker_url,
        model_name=args.model,
        strategy=args.policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
