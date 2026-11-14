"""Which host-device copies wait for queued device work, measured against a fixed 20 ms device sleep.

Each case queues the sleep on one stream, then times one host call made on the stream named in the case.
A call that waits for the sleep takes about 20 ms; one that only enqueues takes well under 1 ms.

usage: python3 sync_semantics.py
"""

import json
import threading
import time

import torch
from sglang.srt.utils.common import async_d2h, async_h2d

DEVICE = torch.device("cuda:0")
SLEEP_MS = 20.0
ROWS = [7, 11, 13, 17]


def cycles_per_ms() -> float:
    probe = 10_000_000
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda._sleep(probe)
    torch.cuda.synchronize()
    return probe / ((time.perf_counter() - start) * 1000)


def timed_call(sleep_stream, call_stream, call, cycles: int) -> float:
    torch.cuda.synchronize()
    with torch.cuda.stream(sleep_stream):
        torch.cuda._sleep(cycles)
    with torch.cuda.stream(call_stream):
        start = time.perf_counter()
        call()
        elapsed = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()
    return elapsed


def cross_thread(shared: bool, cycles: int) -> float:
    """Thread B queues the sleep on the stream thread A then copies on: one shared stream, or each its own."""
    stream_a = torch.cuda.Stream()
    stream_b = stream_a if shared else torch.cuda.Stream()
    queued = threading.Event()
    torch.cuda.synchronize()

    def sleeper() -> None:
        torch.cuda.set_device(DEVICE)
        with torch.cuda.stream(stream_b):
            torch.cuda._sleep(cycles)
        queued.set()

    thread = threading.Thread(target=sleeper)
    thread.start()
    queued.wait()
    with torch.cuda.stream(stream_a):
        start = time.perf_counter()
        torch.tensor(ROWS, device=DEVICE)
        elapsed = (time.perf_counter() - start) * 1000
    thread.join()
    torch.cuda.synchronize()
    return elapsed


def main() -> None:
    torch.cuda.set_device(DEVICE)
    cycles = int(cycles_per_ms() * SLEEP_MS)
    stream = torch.cuda.Stream()
    other = torch.cuda.Stream()
    on_device = torch.arange(4, device=DEVICE)
    pinned = torch.tensor(ROWS).pin_memory()
    pageable_large = torch.zeros(1 << 20)
    torch.cuda.synchronize()
    cases = {
        "torch.tensor(list, device) after sleep on its stream": (
            stream,
            stream,
            lambda: torch.tensor(ROWS, device=DEVICE),
        ),
        "torch.tensor(list, device) after sleep on another stream": (
            other,
            stream,
            lambda: torch.tensor(ROWS, device=DEVICE),
        ),
        "pageable .to(device) after sleep on its stream": (
            stream,
            stream,
            lambda: torch.tensor(ROWS).to(DEVICE),
        ),
        "pageable 4 MB .to(device, non_blocking) after sleep": (
            stream,
            stream,
            lambda: pageable_large.to(DEVICE, non_blocking=True),
        ),
        "pinned .to(device, non_blocking) after sleep on its stream": (
            stream,
            stream,
            lambda: pinned.to(DEVICE, non_blocking=True),
        ),
        "sglang async_h2d after sleep on its stream": (
            stream,
            stream,
            lambda: async_h2d(ROWS, dtype=torch.long, device=DEVICE),
        ),
        ".tolist() after sleep on its stream": (
            stream,
            stream,
            lambda: on_device.tolist(),
        ),
        ".tolist() after sleep on another stream": (
            other,
            stream,
            lambda: on_device.tolist(),
        ),
        "sglang async_d2h after sleep on its stream": (
            stream,
            stream,
            lambda: async_d2h(on_device),
        ),
    }
    results = {}
    for name, (sleep_stream, call_stream, call) in cases.items():
        timed_call(sleep_stream, call_stream, call, cycles)
        results[name] = round(
            min(timed_call(sleep_stream, call_stream, call, cycles) for _ in range(3)),
            3,
        )
    cross_thread(True, cycles)
    results["thread A copy while thread B's sleep is on the shared stream"] = round(
        min(cross_thread(True, cycles) for _ in range(3)), 3
    )
    results["thread A copy while thread B's sleep is on B's own stream"] = round(
        min(cross_thread(False, cycles) for _ in range(3)), 3
    )
    print(json.dumps({"sleep_ms": SLEEP_MS, "host_ms": results}, indent=2))


if __name__ == "__main__":
    main()
