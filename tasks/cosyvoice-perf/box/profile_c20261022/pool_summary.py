"""The memory pass per cell (PLAN.md 4), from the pool probe's lines and the driver's cells.tsv: the Flow prefix pool's
capacity, the most frames it held in the cell's run (one second samples), the failed allocations in the run (rows that
ran their whole history that hop), and the largest whole utterance HiFT decode so far, with its rows and frames.
usage: python3 pool_summary.py <memory pass dir>"""

import json
import sys
from pathlib import Path

GIB = 2**30
BYTES_PER_FRAME = 901120


def main() -> None:
    out = Path(sys.argv[1])
    samples = sorted(
        (json.loads(line) for path in out.glob("pool.*.jsonl") for line in open(path)), key=lambda row: row["t"]
    )
    print("| cell | capacity frames (GB) | most frames held (GB) | failed allocations | HiFT decode peak GB | rows, longest, frames |")
    print("| --- | --- | --- | --- | --- | --- |")
    for line in open(out / "cells.tsv"):
        cell, start, end = line.split()
        window = [row for row in samples if float(start) <= row["t"] <= float(end)]
        before = [row for row in samples if row["t"] < float(start)]
        if not window:
            print(f"| {cell} | no samples | | | | |")
            continue
        last = window[-1]
        failed = last["failed_allocations"] - (before[-1]["failed_allocations"] if before else 0)
        held = max(row["allocated_frames"] for row in window)
        print(
            f"| {cell} | {last['capacity_frames']} ({last['capacity_frames'] * BYTES_PER_FRAME / GIB:.1f}) | "
            f"{held} ({held * BYTES_PER_FRAME / GIB:.1f}) | {failed} | {last['hift_peak_bytes'] / GIB:.2f} | "
            f"{last['hift_peak_rows']}, {last['hift_peak_longest']}, {last['hift_peak_frames']} |"
        )


if __name__ == "__main__":
    main()
