"""SHM relay puts per stage process from shm_put_size_probe records: count, size buckets around
the put's 64 KiB torch threshold, median bytes, and median and p90 put wall time per bucket.

usage: python shm_put_sizes.py RECORD_DIR
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BUCKETS = (
    ("under 64 KiB", 0, 65536),
    ("64 KiB to 1 MiB", 65536, 1 << 20),
    ("1 MiB to 8 MiB", 1 << 20, 8 << 20),
    ("8 MiB and up", 8 << 20, 1 << 62),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    records = [
        json.loads(line)
        for path in sorted(Path(sys.argv[1]).glob("put_*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    processes = sorted({record["process"] for record in records})
    print(f"{len(records)} puts")
    for process in processes:
        mine = [record for record in records if record["process"] == process]
        print(f"{process}: {len(mine)} puts")
        for label, low, high in BUCKETS:
            rows = [record for record in mine if low <= record["bytes"] < high]
            if not rows:
                continue
            else:
                pass
            sizes = [record["bytes"] for record in rows]
            walls = [record["wall_us"] for record in rows]
            print(
                f"  {label:16s} n={len(rows):6d} bytes median {statistics.median(sizes):12.0f}"
                f" min {min(sizes):10d} max {max(sizes):11d}"
                f"  wall median {statistics.median(walls):9.1f} us p90 {percentile(walls, 0.9):9.1f} us"
            )


if __name__ == "__main__":
    main()
