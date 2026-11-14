"""Whether an event's first record waits while another thread blocks in the device runtime.

Thread B blocks in a stream synchronize on a 50 ms device sleep; thread A meanwhile records a fresh
event, and an event it already recorded once, on its own stream. A record that waits takes about
as long as B's remaining block.

usage: python3 event_first_record.py
"""

import json
import threading
import time

import torch

DEVICE = torch.device("cuda:0")
BLOCK_MS = 50.0


def cycles_per_ms() -> float:
    probe = 10_000_000
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda._sleep(probe)
    torch.cuda.synchronize()
    return probe / ((time.perf_counter() - start) * 1000)


def record_while_blocked(event_factory, cycles: int) -> float:
    blocked_stream = torch.cuda.Stream()
    own_stream = torch.cuda.Stream()
    event = event_factory()
    torch.cuda.synchronize()
    entered = threading.Event()

    def blocker() -> None:
        torch.cuda.set_device(DEVICE)
        with torch.cuda.stream(blocked_stream):
            torch.cuda._sleep(cycles)
        entered.set()
        blocked_stream.synchronize()

    thread = threading.Thread(target=blocker)
    thread.start()
    entered.wait()
    time.sleep(0.005)
    with torch.cuda.stream(own_stream):
        start = time.perf_counter()
        event.record()
        elapsed = (time.perf_counter() - start) * 1000
    thread.join()
    torch.cuda.synchronize()
    return elapsed


def recorded_once() -> torch.cuda.Event:
    event = torch.cuda.Event()
    event.record()
    torch.cuda.synchronize()
    return event


def main() -> None:
    torch.cuda.set_device(DEVICE)
    cycles = int(cycles_per_ms() * BLOCK_MS)
    results = {
        "fresh event, first record": [
            round(record_while_blocked(torch.cuda.Event, cycles), 3) for _ in range(3)
        ],
        "event recorded before, record again": [
            round(record_while_blocked(recorded_once, cycles), 3) for _ in range(3)
        ],
    }
    print(json.dumps({"block_ms": BLOCK_MS, "record_host_ms": results}, indent=2))


if __name__ == "__main__":
    main()
