"""Per-step hop timing from hop_probe records, split into the first and second pass.

A request belongs to the second pass when its first record starts at or after the boundary
(the second arm's start, UTC HH:MM:SS on the run's date, e.g. from progress.txt). For every
(process, function, pass): calls, median and p90 wall ms, median thread CPU ms (sync only).

usage: python hop_report.py <hop dir> <YYYY-MM-DD> <HH:MM:SS boundary>
"""

from __future__ import annotations

import collections
import datetime
import glob
import json
import os
import statistics
import sys


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    boundary = (
        datetime.datetime.fromisoformat(
            f"{sys.argv[2]}T{sys.argv[3]}+00:00"
        ).timestamp()
        * 1e9
    )
    records = []
    for path in glob.glob(os.path.join(sys.argv[1], "hop_*.jsonl")):
        with open(path) as handle:
            records.extend(json.loads(line) for line in handle)
    first_seen: dict[str, int] = {}
    for item in records:
        request_id = item["request_id"]
        if request_id is not None:
            first_seen[request_id] = min(
                first_seen.get(request_id, item["start_ns"]), item["start_ns"]
            )
        else:
            pass
    groups: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    for item in records:
        started = first_seen.get(item["request_id"], item["start_ns"])
        pass_name = "2nd" if started >= boundary else "1st"
        groups[(item["process"], item["function"], pass_name)].append(item)
    print(
        f"{'process':24s} {'function':46s} pass {'n':>4s} {'wall med':>9s} "
        f"{'wall p90':>9s} {'cpu med':>9s}"
    )
    for (process, function, pass_name), items in sorted(groups.items()):
        walls = [item["wall_ms"] for item in items]
        cpus = [item["cpu_ms"] for item in items if item["cpu_ms"] is not None]
        cpu = f"{statistics.median(cpus):9.2f}" if cpus else f"{'-':>9s}"
        print(
            f"{process:24s} {function:46s} {pass_name:4s} {len(items):4d} "
            f"{statistics.median(walls):9.2f} {percentile(walls, 0.9):9.2f} {cpu}"
        )


if __name__ == "__main__":
    main()
