# SPDX-License-Identifier: Apache-2.0
"""Does a thread waiting for the GPU on the legacy default stream throttle another thread's kernel
launches on its own stream? #2625 measured it on Blackwell (driver 580.126.20, torch 2.13); this checks the
H100 box. The AR (omni's loop on PyTorch's default stream, the legacy stream) reads every decode step's
token ids with .tolist(), and the vocoder thread launches its kernels from Python on its own stream.

Thread A queues about 2 ms of GPU work (one 1-block sleep kernel) then waits in one of:
  tolist          ids.tolist() on the legacy default stream (what collect_tokens does)
  sync_tolist     torch.cuda.current_stream().synchronize(), then ids.tolist()
  pinned_event    copy_ into a pinned buffer with non_blocking=True, then an event's synchronize()
  own_stream      ids.tolist() with A on a non-default stream of its own
Thread B launches one small kernel at a time on its own stream (synchronized every 64) and counts launches.
Per mode: B's launches per second while A runs, and A's wait per round.

    python legacy_stream_probe_c20261011.py
"""

from __future__ import annotations

import threading
import time

import torch

ROUNDS = 200
WORK_MS = 2.0


def sleep_cycles(milliseconds: float) -> int:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    end.synchronize()
    return int(10_000_000 * milliseconds / start.elapsed_time(end))


def main() -> None:
    torch.cuda.set_device(0)
    cycles = sleep_cycles(WORK_MS)
    for mode in ("tolist", "sync_tolist", "pinned_event", "own_stream", "tolist"):
        launches = 0
        stop = threading.Event()

        def launcher() -> None:
            nonlocal launches
            stream = torch.cuda.Stream()
            counter = torch.zeros(1, device="cuda")
            with torch.cuda.stream(stream):
                while not stop.is_set():
                    for _ in range(64):
                        counter.add_(1)
                        launches += 1
                    stream.synchronize()

        def waiter(waits: list[float]) -> None:
            stream = torch.cuda.Stream() if mode == "own_stream" else torch.cuda.default_stream()
            ids = torch.zeros(32, dtype=torch.long, device="cuda")
            pinned = torch.empty(32, dtype=torch.long, pin_memory=True)
            event = torch.cuda.Event()
            with torch.cuda.stream(stream):
                for _ in range(ROUNDS):
                    torch.cuda._sleep(cycles)
                    ids.add_(1)
                    started = time.perf_counter()
                    if mode == "sync_tolist":
                        stream.synchronize()
                        ids.tolist()
                    elif mode == "pinned_event":
                        pinned.copy_(ids, non_blocking=True)
                        event.record()
                        event.synchronize()
                        pinned.tolist()
                    else:
                        ids.tolist()
                    waits.append(time.perf_counter() - started)

        thread_b = threading.Thread(target=launcher)
        thread_b.start()
        time.sleep(0.3)
        waits: list[float] = []
        thread_a = threading.Thread(target=waiter, args=(waits,))
        before, started = launches, time.perf_counter()
        thread_a.start()
        thread_a.join()
        elapsed = time.perf_counter() - started
        during = launches - before
        stop.set()
        thread_b.join()
        print(
            f"{mode:13s} B launches per second during A {during / elapsed:9.0f} | "
            f"A wait per round {sum(waits) / len(waits) * 1e3:.2f} ms over {elapsed:.2f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
