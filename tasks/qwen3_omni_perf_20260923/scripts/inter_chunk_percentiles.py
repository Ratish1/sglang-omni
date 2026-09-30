"""Inter-chunk gap distribution of SeedTTS boots from speed_results.json per_request records:
count, mean, p50, p90, p95, p99, max, and the share of gaps over 0.5 s and 1 s, split into each
request's first gap (first chunk to second) and the later ones.

usage: python3 inter_chunk_percentiles.py <boot dir> [<boot dir> ...]   (dirs holding seedtts_en/)
"""

from __future__ import annotations

import json
import os
import statistics
import sys


def percentile(ordered: list[float], q: float) -> float:
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def describe(label: str, gaps: list[float]) -> str:
    ordered = sorted(gaps)
    return (
        f"{label:<6} n {len(ordered):5d} mean {statistics.fmean(ordered):.4f} p50 "
        f"{percentile(ordered, 0.5):.4f} p90 {percentile(ordered, 0.9):.4f} p95 "
        f"{percentile(ordered, 0.95):.4f} p99 {percentile(ordered, 0.99):.4f} max "
        f"{ordered[-1]:.3f} >0.5s {sum(g > 0.5 for g in ordered) / len(ordered) * 100:.2f} % "
        f">1s {sum(g > 1.0 for g in ordered) / len(ordered) * 100:.2f} %"
    )


def main() -> None:
    for boot in sys.argv[1:]:
        with open(os.path.join(boot, "seedtts_en", "speed_results.json")) as handle:
            records = json.load(handle)["per_request"]
        first, later = [], []
        for record in records:
            gaps = record.get("inter_chunk_s") or []
            first += gaps[:1]
            later += gaps[1:]
        print(boot)
        print("  " + describe("all", first + later))
        print("  " + describe("first", first))
        print("  " + describe("later", later))


if __name__ == "__main__":
    main()
