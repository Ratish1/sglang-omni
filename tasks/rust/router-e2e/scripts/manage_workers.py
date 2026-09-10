#!/usr/bin/env python3
"""Launch persistent LocalLauncher workers for router comparisons."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from sglang_omni_router.python.launcher.config import (  # noqa: E402
    LocalLauncherConfig,
    load_launcher_config,
)
from sglang_omni_router.python.launcher.local import LocalLauncher  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--model-path")
    parser.add_argument("--model-name")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-gpus-per-worker", type=int, default=1)
    parser.add_argument("--worker-base-port", type=int, default=8011)
    parser.add_argument("--worker-extra-args", default="")
    parser.add_argument("--wait-timeout", type=int, default=600)
    parser.add_argument("--command", default="sgl-omni")
    parser.add_argument("--health-endpoint", default="/health")
    parser.add_argument(
        "--gpu-ids",
        default="0,1",
        help="One comma-separated CUDA assignment per worker (default: 0,1).",
    )
    args = parser.parse_args(argv)
    if args.model_path is not None and args.model_name is None:
        args.model_name = args.model_path
    return args


def main() -> int:
    args = parse_args()
    if args.config is not None:
        config = load_launcher_config(args.config)
    else:
        config = LocalLauncherConfig(
            model_path=args.model_path,
            model_name=args.model_name,
            num_workers=args.num_workers,
            num_gpus_per_worker=args.num_gpus_per_worker,
            worker_base_port=args.worker_base_port,
            worker_extra_args=args.worker_extra_args,
            wait_timeout=args.wait_timeout,
        )
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if len(gpu_ids) != config.num_workers:
        raise ValueError("--gpu-ids must contain one assignment per worker")
    config = config.model_copy(update={"worker_gpu_ids": gpu_ids})
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
