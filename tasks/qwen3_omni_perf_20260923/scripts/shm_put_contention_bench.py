"""A SHM relay put while the same process runs torch compute on another thread, as a stage does.

The PYTHONPATH tree's shm_create_from_tensor is timed on the main thread (fresh segment each
time, median and p99 in microseconds) while a background thread runs a bicubic resize of a video
batch in a loop on torch's intra-op pool, the shape of the Qwen3-Omni preprocessing processor's
work. The background thread's iterations per second are reported for the same window, with and
without puts, so contention shows in both directions.

usage: python shm_put_contention_bench.py --sizes 65536 1048576 8388608 --seconds 20
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import torch
import torch.nn.functional as functional

from sglang_omni.relay.shm import shm_create_from_tensor


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def compute_loop(stop: threading.Event, counter: list[int]) -> None:
    frames = torch.rand(16, 3, 364, 644)
    while not stop.is_set():
        functional.interpolate(
            frames, size=(352, 640), mode="bicubic", antialias=True, align_corners=False
        )
        counter[0] += 1


def compute_rate(seconds: float) -> float:
    stop, counter = threading.Event(), [0]
    worker = threading.Thread(target=compute_loop, args=(stop, counter))
    worker.start()
    time.sleep(seconds)
    stop.set()
    worker.join()
    return counter[0] / seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    print(f"torch threads {torch.get_num_threads()}")
    print(f"compute alone: {compute_rate(args.seconds / 2):.2f} resizes/s")
    for size in args.sizes:
        tensor = torch.randint(0, 255, (size,), dtype=torch.uint8)
        stop, counter = threading.Event(), [0]
        worker = threading.Thread(target=compute_loop, args=(stop, counter))
        worker.start()
        time.sleep(1.0)
        counter[0] = 0
        started = time.perf_counter()
        puts: list[float] = []
        while time.perf_counter() - started < args.seconds:
            begin = time.perf_counter_ns()
            block = shm_create_from_tensor(tensor)
            puts.append((time.perf_counter_ns() - begin) / 1e3)
            block.close()
            block.unlink()
        elapsed = time.perf_counter() - started
        stop.set()
        worker.join()
        print(
            f"{size:>10d} bytes: put median {statistics.median(puts):10.1f} us "
            f"p99 {percentile(puts, 0.99):10.1f} us ({len(puts)} puts); "
            f"compute {counter[0] / elapsed:.2f} resizes/s"
        )


if __name__ == "__main__":
    main()
