"""Where a small SHM relay put spends its time, part by part, and in bursts and concurrency.

Times, over many repetitions with fresh segments each time (median and p99 in microseconds):
the segment's create plus close and unlink alone; posix_fallocate alone; the old numpy
assignment; torch.frombuffer alone; torch copy_ into a frombuffer view; a memoryview slice
assignment (one memcpy through the buffer protocol); and whole puts: main's (numpy), the
branch's (fallocate then torch), and candidate variants. Then the same puts in back-to-back
bursts of 16 and from 8 threads at once, so a per-call cost that grows under repetition or
contention shows.

usage: python shm_put_small_bench.py --sizes 64 1024 16384 65536 1048576 --repeats 3000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import statistics
import time
from multiprocessing import shared_memory

import numpy as np
import torch


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def create(size: int) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(create=True, size=max(size, 1))


def release(shm: shared_memory.SharedMemory) -> None:
    shm.close()
    shm.unlink()


def put_main(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    t_np = tensor.numpy().reshape(-1)
    shm = create(t_np.nbytes)
    view = np.ndarray(t_np.shape, dtype=t_np.dtype, buffer=shm.buf)
    view[:] = t_np[:]
    return shm


def put_branch(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    source = tensor.contiguous().view(torch.uint8).reshape(-1)
    size = source.numel()
    shm = create(size)
    os.posix_fallocate(shm._fd, 0, size)
    torch.frombuffer(shm.buf, dtype=torch.uint8, count=size).copy_(source)
    return shm


def put_fallocate_memoryview(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    source = tensor.contiguous().view(torch.uint8).reshape(-1)
    size = source.numel()
    shm = create(size)
    os.posix_fallocate(shm._fd, 0, size)
    shm.buf[:size] = source.numpy().data
    return shm


def put_fallocate_numpy(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    t_np = tensor.numpy().reshape(-1)
    shm = create(t_np.nbytes)
    os.posix_fallocate(shm._fd, 0, t_np.nbytes)
    view = np.ndarray(t_np.shape, dtype=t_np.dtype, buffer=shm.buf)
    view[:] = t_np[:]
    return shm


def put_torch_without_reservation(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    source = tensor.contiguous().view(torch.uint8).reshape(-1)
    size = source.numel()
    shm = create(size)
    torch.frombuffer(shm.buf, dtype=torch.uint8, count=size).copy_(source)
    return shm


PUTS = {
    "put main (numpy)": put_main,
    "put branch (fallocate + torch)": put_branch,
    "put torch, no reservation": put_torch_without_reservation,
    "put fallocate + memoryview": put_fallocate_memoryview,
    "put fallocate + numpy": put_fallocate_numpy,
}


def time_parts(size: int, repeats: int) -> dict[str, list[float]]:
    tensor = torch.randint(0, 255, (size,), dtype=torch.uint8)
    rows: dict[str, list[float]] = {}

    def record(label: str, start: int) -> None:
        rows.setdefault(label, []).append((time.perf_counter_ns() - start) / 1e3)

    for _ in range(repeats):
        start = time.perf_counter_ns()
        shm = create(size)
        record("create segment", start)
        start = time.perf_counter_ns()
        os.posix_fallocate(shm._fd, 0, size)
        record("posix_fallocate", start)
        start = time.perf_counter_ns()
        view = np.ndarray((size,), dtype=np.uint8, buffer=shm.buf)
        view[:] = tensor.numpy()
        del view
        record("numpy assign (pages reserved)", start)
        start = time.perf_counter_ns()
        destination = torch.frombuffer(shm.buf, dtype=torch.uint8, count=size)
        record("torch.frombuffer", start)
        start = time.perf_counter_ns()
        destination.copy_(tensor)
        record("torch copy_ (pages reserved)", start)
        del destination
        start = time.perf_counter_ns()
        shm.buf[:size] = tensor.numpy().data
        record("memoryview assign (pages reserved)", start)
        start = time.perf_counter_ns()
        release(shm)
        record("close and unlink", start)
        for label, put in PUTS.items():
            start = time.perf_counter_ns()
            shm = put(tensor)
            record(label, start)
            release(shm)
    return rows


def time_single(size: int, repeats: int) -> dict[str, list[float]]:
    tensor = torch.randint(0, 255, (size,), dtype=torch.uint8)
    rows: dict[str, list[float]] = {}
    for _ in range(repeats):
        for label, put in PUTS.items():
            start = time.perf_counter_ns()
            shm = put(tensor)
            rows.setdefault(label, []).append((time.perf_counter_ns() - start) / 1e3)
            release(shm)
    return rows


def time_bursts(size: int, repeats: int, burst: int) -> dict[str, list[float]]:
    tensor = torch.randint(0, 255, (size,), dtype=torch.uint8)
    rows: dict[str, list[float]] = {}
    for _ in range(max(repeats // burst, 1)):
        for label, put in PUTS.items():
            start = time.perf_counter_ns()
            blocks = [put(tensor) for _ in range(burst)]
            rows.setdefault(f"{label}, burst of {burst} (per put)", []).append(
                (time.perf_counter_ns() - start) / 1e3 / burst
            )
            for shm in blocks:
                release(shm)
    return rows


def time_threads(size: int, repeats: int, threads: int) -> dict[str, list[float]]:
    tensor = torch.randint(0, 255, (size,), dtype=torch.uint8)
    rows: dict[str, list[float]] = {}

    def one(put) -> float:
        start = time.perf_counter_ns()
        shm = put(tensor)
        elapsed = (time.perf_counter_ns() - start) / 1e3
        release(shm)
        return elapsed

    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        for label, put in PUTS.items():
            times = list(pool.map(lambda _: one(put), range(repeats)))
            rows[f"{label}, {threads} threads"] = times
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3000)
    parser.add_argument(
        "--tree-put-only",
        action="store_true",
        help="time only the PYTHONPATH tree's own shm_create_from_tensor, in every scenario",
    )
    args = parser.parse_args()
    if args.tree_put_only:
        from sglang_omni.relay.shm import shm_create_from_tensor

        PUTS.clear()
        PUTS["put tree"] = shm_create_from_tensor
    else:
        pass
    print(f"torch threads {torch.get_num_threads()}")
    for size in args.sizes:
        print(f"\n{size} bytes")
        for group in (
            (
                time_single(size, args.repeats)
                if args.tree_put_only
                else time_parts(size, args.repeats)
            ),
            time_bursts(size, args.repeats, 16),
            time_threads(size, args.repeats, 8),
        ):
            for label, values in group.items():
                print(
                    f"  {label:52s} median {statistics.median(values):9.2f} us"
                    f"   p99 {percentile(values, 0.99):9.2f} us"
                )


if __name__ == "__main__":
    main()
