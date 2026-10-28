"""Where torch's copy starts to beat numpy's in a SHM relay put, around torch's parallel grain.

The branch's two put paths (reserve the pages with posix_fallocate, then numpy assignment or
torch frombuffer().copy_) are timed in one process, alternating put by put on fresh segments, at
sizes around at::internal::GRAIN_SIZE (32768 bytes, above which torch splits a byte copy across
its intra-op threads). Each size runs alone and then beside a background thread running the
preprocessing processor's bicubic resize on torch's intra-op pool. Median and p99 in microseconds.

usage: python shm_put_band_bench.py --sizes 32768 32769 49152 65536 --seconds 10
"""

from __future__ import annotations

import argparse
import os
import statistics
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import torch
import torch.nn.functional as functional


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def reserved_segment(size: int) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(create=True, size=size)
    os.posix_fallocate(shm._fd, 0, size)
    return shm


def put_numpy(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    t_np = tensor.numpy().reshape(-1)
    shm = reserved_segment(t_np.nbytes)
    np.ndarray(t_np.shape, dtype=t_np.dtype, buffer=shm.buf)[:] = t_np
    return shm


def put_torch(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    t_np = tensor.numpy().reshape(-1)
    shm = reserved_segment(t_np.nbytes)
    torch.frombuffer(shm.buf, dtype=torch.uint8, count=t_np.nbytes).copy_(
        tensor.reshape(-1).view(torch.uint8)
    )
    return shm


def compute_loop(stop: threading.Event) -> None:
    frames = torch.rand(16, 3, 364, 644)
    while not stop.is_set():
        functional.interpolate(
            frames, size=(352, 640), mode="bicubic", antialias=True, align_corners=False
        )


def time_alternating(tensor: torch.Tensor, seconds: float) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {"numpy": [], "torch": []}
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        for label, put in (("numpy", put_numpy), ("torch", put_torch)):
            begin = time.perf_counter_ns()
            shm = put(tensor)
            rows[label].append((time.perf_counter_ns() - begin) / 1e3)
            shm.close()
            shm.unlink()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    print(f"torch threads {torch.get_num_threads()}")
    print(
        f"{'size':>10s} {'scenario':14s} {'numpy median':>13s} {'torch median':>13s}"
        f" {'delta':>8s} {'numpy p99':>10s} {'torch p99':>10s}"
    )
    for size in args.sizes:
        tensor = torch.randint(0, 255, (size,), dtype=torch.uint8)
        for scenario in ("alone", "beside compute"):
            stop = threading.Event()
            worker = threading.Thread(target=compute_loop, args=(stop,))
            if scenario == "beside compute":
                worker.start()
                time.sleep(1.0)
            else:
                pass
            rows = time_alternating(tensor, args.seconds)
            if scenario == "beside compute":
                stop.set()
                worker.join()
            else:
                pass
            numpy_median = statistics.median(rows["numpy"])
            torch_median = statistics.median(rows["torch"])
            print(
                f"{size:>10d} {scenario:14s} {numpy_median:13.1f} {torch_median:13.1f}"
                f" {torch_median - numpy_median:+8.1f} {percentile(rows['numpy'], 0.99):10.1f}"
                f" {percentile(rows['torch'], 0.99):10.1f}"
            )


if __name__ == "__main__":
    main()
