"""Each pass of a SHM relay hop timed alone, ours against SGLang's ShmPointerMMData mechanics.

For each payload size: the sender's pack (torch.cat of the payload tensors), our put (a new
SharedMemory and a numpy assignment, sglang_omni/relay/shm.py shm_create_from_tensor), SGLang's
put (a new SharedMemory, posix_fallocate, torch copy_ through torch.frombuffer,
srt/managers/mm_utils.py ShmPointerMMData), and the receiver's three choices: our zero-filled
buffer plus copy (stage_io.read_transfer_buffer with ShmGetOperation), an empty buffer plus copy,
and SGLang's private copy-on-write mapping (mmap ACCESS_COPY, torch.frombuffer), each followed
by one full read of the received bytes so a mapping's page faults are counted. Median of the
repeats, in ms; torch threads as set by the caller.

usage: python shm_relay_bench.py <MiB> [<MiB> ...] --repeats 5
"""

from __future__ import annotations

import argparse
import mmap
import os
import statistics
import time
from multiprocessing import shared_memory

import numpy as np
import torch


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return (time.perf_counter() - start) * 1e3, result


def our_put(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    t_np = tensor.numpy().reshape(-1)
    shm = shared_memory.SharedMemory(create=True, size=max(t_np.nbytes, 1))
    view = np.ndarray(t_np.shape, dtype=t_np.dtype, buffer=shm.buf)
    view[:] = t_np[:]
    return shm


def sglang_put(tensor: torch.Tensor) -> shared_memory.SharedMemory:
    nbytes = tensor.numel() * tensor.element_size()
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    os.posix_fallocate(shm._fd, 0, nbytes)
    torch.frombuffer(shm.buf, dtype=torch.uint8).copy_(
        tensor.view(torch.uint8).reshape(-1)
    )
    return shm


def receive_copy(name: str, nbytes: int, zero: bool) -> torch.Tensor:
    dest = (torch.zeros if zero else torch.empty)(nbytes, dtype=torch.uint8)
    shm = shared_memory.SharedMemory(name=name)
    try:
        source = torch.from_numpy(np.ndarray((nbytes,), dtype=np.uint8, buffer=shm.buf))
        dest.copy_(source)
        del source
    finally:
        shm.close()
    return dest


def receive_mapped(name: str, nbytes: int) -> torch.Tensor:
    shm = shared_memory.SharedMemory(name=name)
    try:
        owned = mmap.mmap(shm._fd, nbytes, access=mmap.ACCESS_COPY)
    finally:
        shm.close()
    return torch.frombuffer(owned, dtype=torch.uint8)


def read_all(tensor: torch.Tensor) -> int:
    return int(tensor.view(torch.int64).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes_mib", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    print(f"torch threads {torch.get_num_threads()}")
    for size_mib in args.sizes_mib:
        nbytes = size_mib << 20
        big = torch.randint(0, 255, (nbytes - 4096,), dtype=torch.uint8)
        small = torch.randint(0, 255, (4096,), dtype=torch.uint8)
        rows: dict[str, list[float]] = {}
        for _ in range(args.repeats):
            ms, packed = timed(lambda: torch.cat([big, small]))
            rows.setdefault("sender pack (torch.cat)", []).append(ms)
            for label, put in (("our put", our_put), ("sglang put", sglang_put)):
                ms, shm = timed(lambda: put(packed))
                rows.setdefault(f"sender {label}", []).append(ms)
                name = shm.name
                shm.close()
                if label == "our put":
                    for receive_label, zero in (
                        ("zeros+copy", True),
                        ("empty+copy", False),
                    ):
                        ms, received = timed(lambda: receive_copy(name, nbytes, zero))
                        rows.setdefault(f"receiver {receive_label}", []).append(ms)
                        ms, _ = timed(lambda: read_all(received))
                        rows.setdefault(
                            f"receiver {receive_label}: first read", []
                        ).append(ms)
                        del received
                else:
                    ms, received = timed(lambda: receive_mapped(name, nbytes))
                    rows.setdefault("receiver private mapping", []).append(ms)
                    ms, _ = timed(lambda: read_all(received))
                    rows.setdefault("receiver private mapping: first read", []).append(
                        ms
                    )
                    del received
                cleanup = shared_memory.SharedMemory(name=name)
                cleanup.close()
                cleanup.unlink()
            del packed
        print(f"\n{size_mib} MiB")
        for label, values in rows.items():
            print(f"  {label:42s} {statistics.median(values):9.1f} ms")


if __name__ == "__main__":
    main()
