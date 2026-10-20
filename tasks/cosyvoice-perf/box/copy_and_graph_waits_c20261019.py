"""Which host calls wait for a CUDA stream's queued work, measured with the stream held busy by a sleep kernel.

A. Pageable host-to-device copies, non_blocking, of one size, issued one after another behind a sleep: the host time of
   each and the bytes issued before the first that waited (the driver's staging memory for pageable copies), against
   the same copies from pinned memory.
B. Graph launches behind a sleep: the same graph launched twice, the second while the first is still queued; two
   different graphs; the host time of each launch.

A call waited when its host time is at least half the sleep; the sleep is about 2 s.
usage: CUDA_VISIBLE_DEVICES=3 python3 copy_and_graph_waits_c20261019.py
"""

from __future__ import annotations

import time

import torch

SLEEP_CYCLES = 4_000_000_000


def sleep_ms() -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    torch.cuda._sleep(SLEEP_CYCLES)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e3


def copies(size_bytes: int, count: int, *, pinned: bool, wait_ms: float) -> None:
    sources = [torch.ones(size_bytes // 4, pin_memory=pinned) for _ in range(count)]
    targets = [torch.empty(size_bytes // 4, device="cuda") for _ in range(count)]
    for source, target in zip(sources[:2], targets[:2]):
        target.copy_(source, non_blocking=True)
    torch.cuda.synchronize()
    torch.cuda._sleep(SLEEP_CYCLES)
    durations = []
    for source, target in zip(sources, targets):
        started = time.perf_counter()
        target.copy_(source, non_blocking=True)
        durations.append((time.perf_counter() - started) * 1e3)
    torch.cuda.synchronize()
    waited = [index for index, value in enumerate(durations) if value >= wait_ms]
    first = waited[0] if waited else None
    issued = "none waited" if first is None else f"first wait at copy {first}, {first * size_bytes / 1024:.0f} KiB before it"
    print(
        f"{'pinned' if pinned else 'pageable'} {size_bytes / 1024:7.0f} KiB x {count}: {issued}; "
        f"waits {len(waited)}, max {max(durations):.1f} ms, p50 {sorted(durations)[count // 2]:.3f} ms"
    )


def graph(cycles: int) -> torch.cuda.CUDAGraph:
    captured = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        torch.cuda._sleep(1000)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(captured):
        torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    return captured


def launches(label: str, first: torch.cuda.CUDAGraph, second: torch.cuda.CUDAGraph, wait_ms: float) -> None:
    torch.cuda.synchronize()
    torch.cuda._sleep(SLEEP_CYCLES)
    durations = []
    for captured in (first, second):
        started = time.perf_counter()
        captured.replay()
        durations.append((time.perf_counter() - started) * 1e3)
    torch.cuda.synchronize()
    print(f"{label}: launch ms {durations[0]:.3f} then {durations[1]:.3f} ({'waited' if durations[1] >= wait_ms else 'no wait'})")


def main() -> None:
    torch.cuda.init()
    full = sleep_ms()
    wait_ms = full / 2
    print(f"sleep {full:.0f} ms, a wait is at least {wait_ms:.0f} ms; driver {torch.cuda.get_device_name()}")
    for size in (32 * 1024, 64 * 1024, 64 * 1024 + 4, 96 * 1024, 256 * 1024, 1024 * 1024):
        copies(size, 64, pinned=False, wait_ms=wait_ms)
    copies(32 * 1024, 512, pinned=False, wait_ms=wait_ms)
    for size in (256 * 1024, 1024 * 1024):
        copies(size, 64, pinned=True, wait_ms=wait_ms)
    one, other = graph(1000), graph(1000)
    launches("same graph twice", one, one, wait_ms)
    launches("two graphs", one, other, wait_ms)
    long_one = graph(SLEEP_CYCLES // 4)
    torch.cuda.synchronize()
    started = time.perf_counter()
    long_one.replay()
    first_ms = (time.perf_counter() - started) * 1e3
    started = time.perf_counter()
    long_one.replay()
    second_ms = (time.perf_counter() - started) * 1e3
    torch.cuda.synchronize()
    print(f"a long graph relaunched while it runs: launch ms {first_ms:.3f} then {second_ms:.3f}")


main()
