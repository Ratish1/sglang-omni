#!/usr/bin/env python3
"""Stage payload reads per stage process inside one events cell's window, from hop probe "rp" records (boots run with
OMNI_HOP_PROBE_PAYLOAD_READS=1).

usage: python3 payload_reads.py --cell c64_events <boot> [<boot> ...]

Per process (named by the stage loops its lag sampler watched): reads, MiB a read (mean, p90, max), read time (mean,
p90, max), the share of reads over 50 MiB and their mean time, and the five largest tensor paths by MiB.
"""

import argparse
import collections
import glob
import json
import os
import statistics


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def report(boot: str, cell: str) -> None:
    with open(os.path.join(boot, cell, "window.json"), encoding="utf-8") as source:
        window = json.load(source)
    start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
    print(f"{boot}:")
    for path in sorted(glob.glob(os.path.join(boot, "probe", "*.jsonl"))):
        reads, loops = [], set()
        with open(path, encoding="utf-8") as source:
            for line in source:
                if '"k": "rp"' not in line and '"k": "lag' not in line:
                    continue
                else:
                    pass
                item = json.loads(line)
                if item["k"] == "rp" and start_ns <= item["ns"] <= end_ns:
                    reads.append(item)
                elif item["k"].startswith("lag"):
                    loops.add(item["loop"])
                else:
                    pass
        if not reads:
            continue
        else:
            pass
        mib = [item["bytes"] / 2**20 for item in reads]
        ms = [(item["ns"] - item["ns0"]) / 1e6 for item in reads]
        large = [m for item, m in zip(reads, ms) if item["bytes"] > 50 * 2**20]
        tensors = collections.Counter()
        for item in reads:
            for tensor_path, tensor_mib in item["tensors"]:
                tensors[tensor_path] = max(tensors[tensor_path], tensor_mib)
        print(
            f"  {','.join(sorted(loops)) or os.path.basename(path)}: {len(reads)} reads, MiB mean {statistics.fmean(mib):.1f} "
            f"p90 {quantile(mib, 0.9):.1f} max {max(mib):.1f}; ms mean {statistics.fmean(ms):.1f} p90 {quantile(ms, 0.9):.1f} "
            f"max {max(ms):.1f}; over 50 MiB {len(large)} reads, mean {statistics.fmean(large) if large else 0:.1f} ms"
        )
        for tensor_path, tensor_mib in tensors.most_common(5):
            print(f"      {tensor_mib:8.1f} MiB  {tensor_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("boots", nargs="+")
    args = parser.parse_args()
    for boot in args.boots:
        report(boot, args.cell)


if __name__ == "__main__":
    main()
