# SPDX-License-Identifier: Apache-2.0
"""Does a thread's device-to-host copy that waits for its stream hold up another thread's kernel
launches, without a profiler?

A vocoder thread queues a 20 ms kernel on its stream (one block, so the device stays free for the
other stream), then waits for it through one of three copies of a HiFT step's samples:
  pageable   samples.cpu(), what hift_group does
  pinned     sglang's async_d2h (pinned host buffer, non_blocking copy), then an event synchronize
  sync_first the stream synchronized, then samples.cpu()
An AR thread meanwhile launches one small kernel at a time on its own stream (synchronized every 64
launches) and times each launch call on the host. Per variant: AR launches, the launch time spent
over 1 ms, and the longest launch.

    python d2h_lock_probe_c20261010.py
"""

from __future__ import annotations

import threading
import time

import torch
from sglang.srt.utils.common import async_d2h

ROUNDS = 40
SLEEP_MS = 20


def sleep_cycles(milliseconds: float) -> int:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    end.synchronize()
    return int(10_000_000 * milliseconds / start.elapsed_time(end))


def main() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cycles = sleep_cycles(SLEEP_MS)
    samples = torch.randn(16, 256 * 480, device=device)
    vocoder_stream, ar_stream = torch.cuda.Stream(), torch.cuda.Stream()
    for variant in ("pageable", "pinned", "sync_first", "pageable"):
        launches: list[int] = []
        stop = threading.Event()

        def ar_loop() -> None:
            counter = torch.zeros(1, device=device)
            with torch.cuda.stream(ar_stream):
                while not stop.is_set():
                    for _ in range(64):
                        started = time.perf_counter_ns()
                        counter.add_(1)
                        launches.append(time.perf_counter_ns() - started)
                    ar_stream.synchronize()

        ar_thread = threading.Thread(target=ar_loop)
        ar_thread.start()
        time.sleep(0.2)
        with torch.cuda.stream(vocoder_stream):
            for _ in range(ROUNDS):
                torch.cuda._sleep(cycles)
                if variant == "pageable":
                    host = samples.cpu()
                elif variant == "pinned":
                    host = async_d2h(samples)
                    copy_done = torch.cuda.Event()
                    copy_done.record()
                    copy_done.synchronize()
                else:
                    vocoder_stream.synchronize()
                    host = samples.cpu()
                assert host.shape == samples.shape
        stop.set()
        ar_thread.join()
        slow = [t for t in launches if t > 1_000_000]
        print(
            f"{variant:10s} rounds {ROUNDS} x {SLEEP_MS} ms | AR launches {len(launches):7d} "
            f"over 1 ms {len(slow):4d} sum {sum(slow) / 1e6:7.1f} ms max {max(launches) / 1e6:6.2f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
