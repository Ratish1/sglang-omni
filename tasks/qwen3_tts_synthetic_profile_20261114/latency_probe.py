"""Cycle counts of the persistent predictor layer's primitives on this GPU.

One warp or one CTA times a dependent chain of each primitive with clock64: a chained and an
independent m16n8k16 bf16 MMA, a chained ldmatrix.x4, an ld.global.cg pointer chase over 16 MB
(L2) and over 1 GB (DRAM) in random 128-byte lines, a store then fence.acq_rel.gpu, a store then
red.release.gpu, a 256-thread named barrier, and a clock64 and a %globaltimer read. The probe runs
twice and reports the second, so L2 holds the 16 MB ring.

Usage: python3 latency_probe.py --out <json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sglang.kernels.jit.utils import load_jit

SOURCE = Path(__file__).resolve().parent / "latency_probe.cuh"
NAMES = (
    "mma chained",
    "mma independent (4 accumulators)",
    "ldmatrix.x4 chained",
    "ld.global.cg chase, 16 MB (L2)",
    "ld.global.cg chase, 1 GB (DRAM)",
    "store + fence.acq_rel.gpu",
    "store + red.release.gpu",
    "bar.sync 256 threads",
    "clock64 read",
    "globaltimer read",
)
LINE_WORDS = 32
REPEATS = 256


def ring(footprint_bytes: int, device: torch.device) -> torch.Tensor:
    lines = footprint_bytes // (LINE_WORDS * 4)
    order = np.random.default_rng(0).permutation(lines).astype(np.int64)
    chain = np.zeros(lines * LINE_WORDS, dtype=np.uint32)
    chain[order * LINE_WORDS] = (np.roll(order, -1) * LINE_WORDS).astype(np.uint32)
    return torch.from_numpy(chain).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    module = load_jit(
        "qwen3_tts_latency_probe",
        cuda_files=[str(SOURCE)],
        cuda_wrappers=[("run", "latency_probe::LatencyProbe::run")],
    )
    cycles = torch.zeros(len(NAMES), dtype=torch.int64, device=device)
    l2_ring = ring(16 * 2**20, device)
    dram_ring = ring(2**30, device)
    scratch = torch.zeros(REPEATS + 64, dtype=torch.uint32, device=device)
    for _ in range(2):
        module.run(cycles, l2_ring, dram_ring, scratch)
        torch.cuda.synchronize()
    results = dict(zip(NAMES, cycles.cpu().tolist()))
    print(json.dumps(results, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
