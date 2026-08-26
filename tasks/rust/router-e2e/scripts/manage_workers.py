#!/usr/bin/env python3
"""Launch exactly two LocalLauncher workers and keep them alive for router A/Bs."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from sglang_omni_router.launcher import LocalLauncher, load_launcher_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--command", default="sgl-omni")
    parser.add_argument("--health-endpoint", default="/health")
    parser.add_argument(
        "--gpu-ids",
        default="0,1",
        help="Two comma-separated physical GPU ids (default: 0,1).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_launcher_config(args.config)
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if len(gpu_ids) != 2:
        raise ValueError("--gpu-ids must contain exactly two GPU ids")
    config = config.model_copy(update={"worker_gpu_ids": gpu_ids})
    if config.num_workers != 2:
        raise ValueError(
            f"qualification requires exactly two workers, got {config.num_workers}"
        )
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    launcher = LocalLauncher(
        config,
        command=args.command,
        health_endpoint=args.health_endpoint,
    )
    try:
        urls = launcher.launch_and_wait()
        print("WORKERS_READY " + " ".join(urls), flush=True)
        while not stop.wait(1.0):
            exited = [
                worker
                for worker in launcher.workers
                if worker.process.poll() is not None
            ]
            if exited:
                statuses = ", ".join(
                    f"{worker.url}={worker.process.returncode}" for worker in exited
                )
                raise RuntimeError(f"managed worker exited unexpectedly: {statuses}")
    finally:
        launcher.shutdown()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    raise SystemExit(main())
