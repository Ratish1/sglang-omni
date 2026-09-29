"""Launch geometry census per owner: how much of the GPU each component's kernels can use.

For every kernel in the bench window, from its launch record alone (no counters):
the SMs its grid can cover (CTAs over SMs, capped at 1), the blocks per SM its registers,
shared memory and block size allow (the Hopper limits: 64 warps, 65,536 registers,
228 KiB shared memory and 32 blocks per SM), and from those the resident warps per SM
it can reach. Weighted by kernel time per owner (owners as in pipeline_census.py), this
bounds the SM active and SM occupancy DCGM reads, and shows which owner leaves SMs empty.

usage: python occupancy_census.py REPORT.sqlite --bench-log bench.log [--sms 132]
"""

from __future__ import annotations

import argparse
import collections
import math

from pipeline_census import Report

MAX_WARPS = 64
MAX_REGISTERS = 65536
MAX_SHARED = 228 * 1024
MAX_BLOCKS = 32
REGISTER_UNIT = 256
BLOCK_SHARED_RESERVED = 1024


def blocks_per_sm(threads: int, registers: int, shared: int) -> int:
    warps = max(1, math.ceil(threads / 32))
    by_warps = MAX_WARPS // warps
    per_warp = math.ceil(max(registers, 1) * 32 / REGISTER_UNIT) * REGISTER_UNIT
    by_registers = MAX_REGISTERS // (per_warp * warps)
    by_shared = MAX_SHARED // (shared + BLOCK_SHARED_RESERVED)
    return max(0, min(MAX_BLOCKS, by_warps, by_registers, by_shared))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log")
    parser.add_argument("--sms", type=int, default=132)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    r = Report(args.report, args.bench_log)
    config = {}
    for start, corr, gx, gy, gz, bx, by, bz, regs, static, dynamic in r.db.execute(
        "select start, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ, "
        "registersPerThread, staticSharedMemory, dynamicSharedMemory "
        "from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?",
        (r.t0, r.t1),
    ):
        config[(start, corr)] = (gx * gy * gz, bx * by * bz, regs, static + dynamic)

    sms = args.sms
    owners = collections.defaultdict(lambda: collections.Counter())
    buckets = collections.defaultdict(lambda: collections.Counter())
    families = collections.defaultdict(lambda: collections.Counter())
    for start, end, owner, name, stream, ident, corr, node in r.device:
        entry = config.get((start, corr))
        if entry is None:
            continue
        ctas, threads, regs, shared = entry
        duration = end - start
        limit = blocks_per_sm(threads, regs, shared)
        warps = max(1, math.ceil(threads / 32))
        coverage = min(1.0, ctas / sms)
        per_sm_blocks = min(limit, math.ceil(ctas / sms)) if limit else 0
        occupancy = coverage * per_sm_blocks * warps / MAX_WARPS
        stats = owners[owner]
        stats["time"] += duration
        stats["kernels"] += 1
        stats["coverage"] += coverage * duration
        stats["occupancy"] += occupancy * duration
        stats["full_occupancy"] += (limit * warps / MAX_WARPS) * duration
        band = (
            "under 10%"
            if coverage < 0.1
            else (
                "10 to 25%"
                if coverage < 0.25
                else (
                    "25 to 50%"
                    if coverage < 0.5
                    else "50 to 99%" if coverage < 1 else "all SMs"
                )
            )
        )
        buckets[owner][band] += duration
        family = families[(owner, name[:56])]
        family["time"] += duration
        family["count"] += 1
        family["coverage"] += coverage * duration
        family["occupancy"] += occupancy * duration
        family["ctas"] += ctas

    total = sum(s["time"] for s in owners.values())
    print(f"kernels by owner, weighted by kernel time; SMs {sms}")
    print(
        f"{'owner':<16}{'kernel s':>9}{'share':>7}{'SM cover':>10}{'warps/SM':>10}{'if full':>9}  time by SM coverage"
    )
    order = ["under 10%", "10 to 25%", "25 to 50%", "50 to 99%", "all SMs"]
    for owner, stats in sorted(owners.items(), key=lambda kv: -kv[1]["time"]):
        t = stats["time"]
        split = " ".join(
            f"{band} {100 * buckets[owner][band] / t:.0f}%"
            for band in order
            if buckets[owner][band]
        )
        print(
            f"{owner:<16}{t / 1e9:>9.2f}{100 * t / total:>6.1f}%{100 * stats['coverage'] / t:>9.1f}%"
            f"{100 * stats['occupancy'] / t:>9.1f}%{100 * stats['full_occupancy'] / t:>8.1f}%  {split}"
        )
    all_cov = sum(s["coverage"] for s in owners.values()) / total
    all_occ = sum(s["occupancy"] for s in owners.values()) / total
    print(
        f"{'all':<16}{total / 1e9:>9.2f}{100.0:>6.1f}%{100 * all_cov:>9.1f}%{100 * all_occ:>9.1f}%"
    )
    print(
        "  SM cover: CTAs over SMs; warps/SM: resident warps over 64 averaged over all SMs;"
    )
    print(
        "  if full: the occupancy the kernel's registers, shared memory and block size allow"
    )
    print(f"\ntop kernel families by time, per owner (top {args.top})")
    for owner in sorted(owners, key=lambda o: -owners[o]["time"]):
        rows = sorted(
            ((k, v) for k, v in families.items() if k[0] == owner),
            key=lambda kv: -kv[1]["time"],
        )
        print(f" {owner}")
        for (o, name), v in rows[: args.top]:
            print(
                f"   {name:<56} {v['time'] / 1e6:>8.1f} ms {v['count']:>8} calls "
                f"{v['ctas'] / v['count']:>7.0f} CTAs  cover {100 * v['coverage'] / v['time']:>5.1f}%  "
                f"warps/SM {100 * v['occupancy'] / v['time']:>5.1f}%"
            )


if __name__ == "__main__":
    main()
