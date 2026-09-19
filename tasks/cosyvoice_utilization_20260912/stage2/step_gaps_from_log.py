#!/usr/bin/env python3
"""Vocoder step gaps from a serve.log.

The streaming vocoder logs one line when a hop step or a final step starts.
The gap to the next such line is the step plus any idle wait before the next
step, so at a saturated concurrency it bounds the step from above. Gaps are
grouped by step kind, by rows, and for a hop cache boot by how the rows split.

  python step_gaps_from_log.py <run dir> [<run dir> ...]
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime

LINE = re.compile(
    r"^(\S+ \S+) \[INFO\] \S+streaming_vocoder: Fun-CosyVoice3 "
    r"(causal|leftover) Flow batch size=(\d+)(?: cached=(\d+))?"
)


def steps_of(path: str) -> list[tuple[float, str, int, int | None]]:
    steps = []
    with open(path) as log:
        for line in log:
            match = LINE.match(line)
            if match is None:
                continue
            stamp, kind, rows, cached = match.groups()
            at = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f").timestamp()
            steps.append((at, kind, int(rows), None if cached is None else int(cached)))
    return steps


def summary(gaps: list[float]) -> str:
    gaps = sorted(gaps)
    p95 = gaps[min(len(gaps) - 1, int(0.95 * len(gaps)))]
    return (
        f"n={len(gaps):4d} sum={sum(gaps):7.1f}s p50={statistics.median(gaps) * 1e3:6.0f}ms "
        f"p95={p95 * 1e3:6.0f}ms max={gaps[-1] * 1e3:6.0f}ms"
    )


def main() -> None:
    for run in sys.argv[1:]:
        steps = steps_of(f"{run}/serve.log")
        print(f"== {run}")
        span = steps[-1][0] - steps[0][0]
        hop_rows = sum(rows for _, kind, rows, _ in steps if kind == "causal")
        final_rows = sum(rows for _, kind, rows, _ in steps if kind == "leftover")
        print(
            f"steps={len(steps)} span={span:.1f}s hop rows={hop_rows} "
            f"final rows={final_rows} hops per final={hop_rows / final_rows:.2f}"
        )
        groups: dict[str, list[float]] = {}
        for (at, kind, rows, cached), (after, *_rest) in zip(steps, steps[1:]):
            gap = after - at
            size = "1-4" if rows <= 4 else "5-9" if rows <= 9 else "10-16"
            groups.setdefault(f"{kind:8s} all", []).append(gap)
            groups.setdefault(f"{kind:8s} rows {size}", []).append(gap)
            if cached is not None:
                split = (
                    "all cached"
                    if cached == rows
                    else "none cached" if cached == 0 else "mixed"
                )
                groups.setdefault(f"{kind:8s} rows {size} {split}", []).append(gap)
                groups.setdefault(f"{kind:8s} {split}", []).append(gap)
        for name in sorted(groups):
            print(f"  {name:36s} {summary(groups[name])}")


if __name__ == "__main__":
    main()
