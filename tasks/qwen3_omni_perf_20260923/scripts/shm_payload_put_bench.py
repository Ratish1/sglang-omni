"""The sender side of a stage payload hop over SHM, packing included, for the PYTHONPATH tree.

Main packs a payload with stage_io.pack_tensors (a torch.cat into a new buffer) and puts it with
shm_create_from_tensor (a second copy into the block); the one-copy put packs with
stage_io.pack_tensor_chunks and concatenates straight into the block in shm_create_from_chunks.
The script times whichever the tree has, on fresh blocks, for a one-tensor payload (float32) and
a mixed payload (bool, float32, int64 and bfloat16 tensors with alignment padding) of each total
size, alone and beside a thread running the preprocessing resize on torch's pool. Median and p99
in microseconds.

usage: python shm_payload_put_bench.py --sizes 1024 1048576 --repeats 200 --seconds-cap 20
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import torch
import torch.nn.functional as functional

from sglang_omni.comm import stage_io
from sglang_omni.relay import shm as shm_relay


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def make_payloads(total_bytes: int) -> dict[str, dict[str, torch.Tensor]]:
    quarter = max(total_bytes // 4, 8)
    return {
        "one tensor": {"features": torch.rand(max(total_bytes // 4, 1))},
        "mixed": {
            "flags": torch.rand(3) > 0.5,
            "features": torch.rand(quarter // 4),
            "ids": torch.randint(0, 1 << 30, (quarter // 8,)),
            "embeds": torch.rand(quarter, dtype=torch.bfloat16),
        },
    }


def send(tensors: dict[str, torch.Tensor]):
    if hasattr(stage_io, "pack_tensor_chunks"):
        chunks, _ = stage_io.pack_tensor_chunks(tensors, device="cpu")
        return shm_relay.shm_create_from_chunks(chunks)
    else:
        packed, _ = stage_io.pack_tensors(tensors, device="cpu")
        return shm_relay.shm_create_from_tensor(packed)


def compute_loop(stop: threading.Event) -> None:
    frames = torch.rand(16, 3, 364, 644)
    while not stop.is_set():
        functional.interpolate(
            frames, size=(352, 640), mode="bicubic", antialias=True, align_corners=False
        )


def time_sends(
    tensors: dict[str, torch.Tensor], repeats: int, seconds_cap: float
) -> list[float]:
    times: list[float] = []
    started = time.perf_counter()
    while len(times) < repeats and time.perf_counter() - started < seconds_cap:
        begin = time.perf_counter_ns()
        block = send(tensors)
        times.append((time.perf_counter_ns() - begin) / 1e3)
        block.close()
        block.unlink()
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seconds-cap", type=float, default=20.0)
    args = parser.parse_args()
    path = "one copy" if hasattr(stage_io, "pack_tensor_chunks") else "pack then put"
    print(f"torch threads {torch.get_num_threads()}, path: {path}")
    for total_bytes in args.sizes:
        for kind, tensors in make_payloads(total_bytes).items():
            for scenario in ("alone", "beside compute"):
                stop = threading.Event()
                worker = threading.Thread(target=compute_loop, args=(stop,))
                if scenario == "beside compute":
                    worker.start()
                    time.sleep(1.0)
                else:
                    pass
                times = time_sends(tensors, args.repeats, args.seconds_cap)
                if scenario == "beside compute":
                    stop.set()
                    worker.join()
                else:
                    pass
                print(
                    f"{total_bytes:>11d} B {kind:10s} {scenario:14s} median"
                    f" {statistics.median(times):11.1f} us p99 {percentile(times, 0.99):11.1f} us"
                    f" (n {len(times)})"
                )


if __name__ == "__main__":
    main()
