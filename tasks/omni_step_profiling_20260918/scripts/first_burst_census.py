"""Per-request first chunk path for the earliest requests of a census run.

The census window opens at the benchmark's "Benchmarking N requests" line, so the first
burst of a closed-loop run (the concurrency's worth of requests sent together) loses its
earliest marks. This reader opens the window PAD_MS earlier and prints, for the first N
requests by arrival at preprocessing, the time of each first chunk segment (the points of
pipeline_census.py section F).

usage: python first_burst_census.py serve.sqlite bench.log [N] [PAD_MS]
"""

from __future__ import annotations

import collections
import re
import sys

import pipeline_census
from nsys_metrics import bench_window

POINT_NAMES = (
    "pre inbox put",
    "pre.payload start",
    "pre.payload end",
    "engine inbox put",
    "build start",
    "admitted",
    "prefill launch",
    "first frame out",
    "vocoder ingest",
    "decode queued",
    "decode taken",
    "first chunk committed",
)


def main() -> None:
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    pad_ns = int(float(sys.argv[4]) * 1e6) if len(sys.argv) > 4 else 3_000_000_000

    def padded_window(db, bench_log):
        start, end = bench_window(db, bench_log)
        return start - pad_ns, end

    pipeline_census.bench_window = padded_window
    report = pipeline_census.Report(sys.argv[1], sys.argv[2])
    first = collections.defaultdict(dict)
    for items in report.ranges.values():
        for item in items:
            rid = item.rid()
            if rid and item.kind not in first[rid]:
                first[rid][item.kind] = item
            else:
                pass
    marks = collections.defaultdict(dict)
    for t, _, label in report.marks:
        match = re.match(r"(q \S+|\S+)(?: \S+)? rid=(\S+)", label)
        if match:
            key = match.group(1)
            if key.startswith("q "):
                key += ":" + label.rsplit("t=", 1)[-1]
            else:
                pass
            marks[match.group(2)].setdefault(key, t)
        else:
            pass
    rows = []
    for rid, kinds in first.items():
        mark = marks.get(rid, {})
        payload = kinds.get("pre.payload")
        commit = kinds.get("voc.commit")
        if payload is None or commit is None:
            continue
        else:
            pass
        points = (
            mark.get("q preprocessing.in.put:new_request"),
            payload.start,
            payload.end,
            mark.get("q tts_engine.in.put:new_request"),
            kinds["sched.build"].start if "sched.build" in kinds else None,
            mark.get("sched.admit"),
            mark.get("sched.prefill"),
            mark.get("q tts_engine.out.put:stream"),
            kinds["voc.ingest"].start if "voc.ingest" in kinds else None,
            mark.get("q voc_initial.put:-", mark.get("voc.put")),
            mark.get("q voc_initial.get:-", mark.get("voc.take")),
            commit.end,
        )
        arrival = points[0] if points[0] is not None else payload.start
        rows.append((arrival, rid, points))
    rows.sort()
    origin = rows[0][0]
    header = "".join(f"{name[:9]:>10}" for name in POINT_NAMES[1:])
    print(
        f"first {count} requests; ms from the previous point (arrival in ms from the first)"
    )
    print(f"{'arrival':>9}{header}{'total':>9}")
    for arrival, _, points in rows[:count]:
        cells = []
        previous = points[0] if points[0] is not None else points[1]
        for point in points[1:]:
            if point is None:
                cells.append(f"{'-':>10}")
            else:
                cells.append(f"{(point - previous) / 1e6:>10.1f}")
                previous = point
        total = (points[-1] - (points[0] or points[1])) / 1e6
        print(f"{(arrival - origin) / 1e6:>9.1f}{''.join(cells)}{total:>9.1f}")


if __name__ == "__main__":
    main()
