# SPDX-License-Identifier: Apache-2.0
"""Deterministic multiprocess CPU interferer for saturation experiments."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import signal
import time
from collections.abc import Sequence


def parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if end < start:
                raise ValueError(f"invalid descending CPU range: {item}")
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(item))
    if len(set(cpus)) != len(cpus):
        raise ValueError("CPU list contains duplicates")
    return cpus


def _burn(stop: multiprocessing.synchronize.Event, cpu: int | None) -> None:
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    value = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    while not stop.is_set():
        for _ in range(100_000):
            value ^= (value << 13) & mask
            value ^= value >> 7
            value ^= (value << 17) & mask


def run(workers: int, cpus: Sequence[int]) -> None:
    if workers < 1:
        raise ValueError("--workers must be positive")
    if cpus and any(cpu not in os.sched_getaffinity(0) for cpu in cpus):
        raise ValueError("requested CPU is outside the parent process affinity")
    context = multiprocessing.get_context("spawn")
    stop = context.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    processes = [
        context.Process(
            target=_burn,
            args=(stop, cpus[index % len(cpus)] if cpus else None),
            name=f"cpu-interferer-{index}",
        )
        for index in range(workers)
    ]
    for process in processes:
        process.start()
    print(
        {
            "parent_pid": os.getpid(),
            "workers": workers,
            "worker_pids": [process.pid for process in processes],
            "cpus": list(cpus),
            "started_wall_ns": time.time_ns(),
        },
        flush=True,
    )
    try:
        while not stop.wait(1.0):
            failed = [
                process for process in processes if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(
                    "CPU interferer worker exited early: "
                    + ", ".join(
                        f"{process.name}={process.exitcode}" for process in failed
                    )
                )
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--cpus",
        default="",
        help="Optional Linux CPU list; workers are assigned round-robin.",
    )
    args = parser.parse_args()
    run(args.workers, parse_cpu_list(args.cpus))


if __name__ == "__main__":
    main()
