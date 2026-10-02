# SPDX-License-Identifier: Apache-2.0
"""Per call GPU kernel time, kernel count and wall span of formal traces, and a per kernel
name diff between two arms of the same case.

    python trace_summary.py --iters 10 <arm dir>...            one row per arm and case
    python trace_summary.py --iters 10 --diff <a dir> <b dir>  kernels whose time moved
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def kernels(trace_dir: Path) -> list[dict]:
    path = next(trace_dir.rglob("*.json.gz"))
    with gzip.open(path) as handle:
        events = json.load(handle)["traceEvents"]
    return [event for event in events if event.get("cat") == "kernel"]


def summary(trace_dir: Path, iters: int) -> tuple[float, float, float]:
    events = kernels(trace_dir)
    busy = sum(event["dur"] for event in events)
    start = min(event["ts"] for event in events)
    end = max(event["ts"] + event["dur"] for event in events)
    return busy / iters / 1000, len(events) / iters, (end - start) / iters / 1000


def by_name(trace_dir: Path, iters: int) -> dict[str, tuple[float, float]]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for event in kernels(trace_dir):
        totals[event["name"][:110]][0] += event["dur"]
        totals[event["name"][:110]][1] += 1
    return {
        name: (dur / iters / 1000, count / iters)
        for name, (dur, count) in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    if args.diff:
        first, second = (Path(d) for d in args.dirs)
        a, b = by_name(first, args.iters), by_name(second, args.iters)
        rows = [
            (b.get(name, (0, 0))[0] - a.get(name, (0, 0))[0], name)
            for name in set(a) | set(b)
        ]
        rows.sort(key=lambda row: -abs(row[0]))
        for delta, name in rows[: args.top]:
            ma, ca = a.get(name, (0, 0))
            mb, cb = b.get(name, (0, 0))
            print(
                f"{ma:8.3f} ms x{ca:5.0f} | {mb:8.3f} ms x{cb:5.0f} | {delta:+8.3f}  {name}"
            )
    else:
        for arm in args.dirs:
            for formal in sorted(Path(arm).glob("*/formal")):
                busy, count, span = summary(formal, args.iters)
                print(
                    f"{Path(arm).name:14s} {formal.parent.name:16s} "
                    f"gpu {busy:8.2f} ms  kernels {count:7.0f}  span {span:8.2f} ms"
                )


if __name__ == "__main__":
    main()
