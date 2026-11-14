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
BLOCKING_CALLS = (
    "stream synchronize",
    "device synchronize",
    "event synchronize",
    "device to host read",
)


def cycles_per_ms() -> float:
    probe = 10_000_000
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda._sleep(probe)
    torch.cuda.synchronize()
    return probe / ((time.perf_counter() - start) * 1000)


def block_in(kind: str, stream: torch.cuda.Stream) -> None:
    if kind == "stream synchronize":
        stream.synchronize()
    elif kind == "device synchronize":
        torch.cuda.synchronize()
    elif kind == "event synchronize":
        event = torch.cuda.Event()
        event.record(stream)
        event.synchronize()
    else:
        with torch.cuda.stream(stream):
            torch.ones(1, device=DEVICE).tolist()


def record_while_blocked(event_factory, cycles: int, kind: str) -> float:
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
        block_in(kind, blocked_stream)

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
    results = {}
    for kind in BLOCKING_CALLS:
        for label, factory in (
            ("fresh event, first record", torch.cuda.Event),
            ("event recorded before, record again", recorded_once),
        ):
            results[f"{kind}: {label}"] = [
                round(record_while_blocked(factory, cycles, kind), 3) for _ in range(3)
            ]
    print(json.dumps({"block_ms": BLOCK_MS, "record_host_ms": results}, indent=2))


if __name__ == "__main__":
    main()
