"""The page reservation's cost on a fresh SHM block, SGLang's own creation against the tree's.

Variants, each creating a fresh block for one CPU tensor and closing it, as one put does:

- sglang: SGLang's installed ShmPointerMMData(tensor), whose creation reserves the pages with
  posix_fallocate on Linux and copies with torch.frombuffer(...).copy_.
- sglang_noreserve: the same creation line for line without the posix_fallocate call.
- sglang_split: the reserving creation with the reservation and the copy timed apart.
- tree: the tree's shm_create_from_chunks over [tensor], then close.

The block is unlinked after the timed region. Variants rotate their order every iteration. For
every size and torch thread count it prints the median and p90 wall time over ITERATIONS after
WARMUP, and checks every block's bytes against the source once per variant and size.

usage: python shm_reserve_bench.py OUT_JSON [ITERATIONS] [WARMUP]
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from multiprocessing import shared_memory

import torch
from sglang.srt.managers.mm_utils import ShmPointerMMData

from sglang_omni.relay.shm import shm_create_from_chunks

SIZES = (225_000, 1_048_576, 8_388_608, 67_108_864, 231_000_000, 346_030_692)
THREADS = (1, 8, 28)
VARIANTS = ("sglang", "sglang_noreserve", "sglang_split", "tree")


def sglang_creation(tensor: torch.Tensor, reserve: bool) -> tuple[str, float, float]:
    nbytes = tensor.numel() * tensor.element_size()
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    reserve_s = 0.0
    try:
        start = time.perf_counter()
        if reserve:
            os.posix_fallocate(shm._fd, 0, nbytes)
        else:
            pass
        reserve_s = time.perf_counter() - start
        start = time.perf_counter()
        dst = torch.frombuffer(shm.buf, dtype=torch.uint8)
        dst.copy_(tensor.view(torch.uint8).reshape(-1))
        copy_s = time.perf_counter() - start
    except BaseException:
        shm.close()
        shm.unlink()
        raise
    name = shm.name
    shm.close()
    return name, reserve_s, copy_s


def run_variant(variant: str, tensor: torch.Tensor) -> tuple[str, float, float, float]:
    start = time.perf_counter()
    reserve_s = 0.0
    copy_s = 0.0
    if variant == "sglang":
        name = ShmPointerMMData(tensor).shm_name
    elif variant == "sglang_noreserve":
        name, reserve_s, copy_s = sglang_creation(tensor, reserve=False)
    elif variant == "sglang_split":
        name, reserve_s, copy_s = sglang_creation(tensor, reserve=True)
    else:
        shm = shm_create_from_chunks([tensor])
        name = shm.name
        shm.close()
    return name, time.perf_counter() - start, reserve_s, copy_s


def check_and_unlink(name: str, tensor: torch.Tensor, check: bool) -> None:
    shm = shared_memory.SharedMemory(name=name)
    try:
        if check:
            block = torch.frombuffer(shm.buf, dtype=torch.uint8)[: tensor.numel()]
            if not torch.equal(block, tensor):
                raise AssertionError(f"block {name} differs from its source")
            else:
                pass
            del block
        else:
            pass
    finally:
        shm.close()
        shm.unlink()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    out_path = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    generator = torch.Generator().manual_seed(0)
    mounts = subprocess.run(["mount"], capture_output=True, text=True).stdout
    environment = {
        "kernel": platform.release(),
        "dev_shm_mount": [line for line in mounts.splitlines() if " /dev/shm " in line],
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__,
    }
    print(json.dumps(environment))
    rows = []
    for threads in THREADS:
        torch.set_num_threads(threads)
        for size_bytes in SIZES:
            tensor = torch.randint(
                0, 256, (size_bytes,), dtype=torch.uint8, generator=generator
            )
            times: dict[str, list[float]] = {variant: [] for variant in VARIANTS}
            reserves: list[float] = []
            copies: list[float] = []
            copies_noreserve: list[float] = []
            for iteration in range(warmup + iterations):
                shift = iteration % len(VARIANTS)
                for variant in VARIANTS[shift:] + VARIANTS[:shift]:
                    name, total_s, reserve_s, copy_s = run_variant(variant, tensor)
                    check_and_unlink(name, tensor, check=iteration == 0)
                    if iteration >= warmup:
                        times[variant].append(total_s * 1000)
                        if variant == "sglang_split":
                            reserves.append(reserve_s * 1000)
                            copies.append(copy_s * 1000)
                        elif variant == "sglang_noreserve":
                            copies_noreserve.append(copy_s * 1000)
                        else:
                            pass
                    else:
                        pass
            row = {
                "threads": threads,
                "size_bytes": size_bytes,
                "median_ms": {
                    variant: statistics.median(values)
                    for variant, values in times.items()
                },
                "p90_ms": {
                    variant: percentile(values, 0.9)
                    for variant, values in times.items()
                },
                "split_reserve_median_ms": statistics.median(reserves),
                "split_copy_median_ms": statistics.median(copies),
                "noreserve_copy_median_ms": statistics.median(copies_noreserve),
            }
            rows.append(row)
            medians = row["median_ms"]
            print(
                f"threads {threads:>2} size {size_bytes:>11,}: "
                + " ".join(f"{variant} {medians[variant]:.3f}" for variant in VARIANTS)
                + f" | reserve {row['split_reserve_median_ms']:.3f} copy after reserve "
                f"{row['split_copy_median_ms']:.3f} copy without {row['noreserve_copy_median_ms']:.3f} ms",
                flush=True,
            )
    with open(out_path, "w") as handle:
        json.dump({"environment": environment, "rows": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
