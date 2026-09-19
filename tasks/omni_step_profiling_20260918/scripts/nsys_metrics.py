"""GPU metrics of an nsys profile sqlite export, over one window (run on the box).

Window: with --bench-log, the benchmark's timed requests ("Benchmarking N requests" to
"Results saved" in the log, converted to session time through the session start's local
time); without it, first to last kernel. Prints the window means of the GPU metrics
(zeros included), the device time per launching thread, then attributes every metric
sample to the kernels executing at its timestamp (kernels of different streams may
overlap; a sample counts for each) and prints, per kernel name ranked by device time:
calls, total ms and the mean of each metric over its samples. A metric sample is the
mean of its sampling period, so per-kernel means are trustworthy only for kernels or
runs of one kernel that are long against the period.

usage: python nsys_metrics.py REPORT.sqlite [--bench-log bench.log] [--top 30]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics
from datetime import datetime

METRICS = (
    "GR Active",
    "SM Issue",
    "SMs Active",
    "Tensor Active",
    "Compute Warps in Flight",
    "DRAM Read Bandwidth",
    "DRAM Write Bandwidth",
)
STAMP = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)\s+.*"


def bench_window(db: sqlite3.Connection, bench_log: str) -> tuple[int, int]:
    start = end = None
    with open(bench_log, errors="replace") as handle:
        lines = handle.readlines()
    for line in lines:
        begun = re.match(STAMP + r"Benchmarking (\d+) requests", line)
        saved = re.match(STAMP + r"Results saved to", line)
        if begun:
            start = datetime.fromisoformat(
                f"{begun.group(1).replace(' ', 'T')}.{begun.group(2)}"
            )
        elif saved and start is not None:
            end = datetime.fromisoformat(
                f"{saved.group(1).replace(' ', 'T')}.{saved.group(2)}"
            )
            break
    if start is None or end is None:
        raise SystemExit(f"no Benchmarking / Results saved pair in {bench_log}")
    session = datetime.fromisoformat(
        db.execute("select localTime from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    )
    return int((start - session).total_seconds() * 1e9), int(
        (end - session).total_seconds() * 1e9
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    if args.bench_log:
        t0, t1 = bench_window(db, args.bench_log)
    else:
        t0, t1 = db.execute(
            "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
    kernels = db.execute(
        "select start, end, deviceId, demangledName, correlationId, globalPid from CUPTI_ACTIVITY_KIND_KERNEL "
        "where end > ? and start < ? order by start",
        (t0, t1),
    ).fetchall()
    if not kernels:
        print("no kernel activity in the window")
        return
    device = collections.Counter(k[2] for k in kernels).most_common(1)[0][0]
    kernels = [k for k in kernels if k[2] == device]
    print(
        f"window {t0 / 1e9:.3f} .. {t1 / 1e9:.3f} s of the session ({(t1 - t0) / 1e9:.3f} s), device {device}, {len(kernels)} kernels"
    )

    merged: list[list[int]] = []
    for start, end, *_ in kernels:
        start, end = max(start, t0), min(end, t1)
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    busy_union = sum(hi - lo for lo, hi in merged)
    print(
        f"kernel union {busy_union / 1e6:.1f} ms = {100 * busy_union / (t1 - t0):.1f}% of the window"
    )

    ids = {}
    for metric_id, name in db.execute(
        "select metricId, metricName from TARGET_INFO_GPU_METRICS"
    ):
        for metric in METRICS:
            if str(name).startswith(metric):
                ids[metric] = metric_id
    samples = {
        metric: db.execute(
            "select timestamp, value from GPU_METRICS where metricId = ? and timestamp between ? and ? order by timestamp",
            (metric_id, t0, t1),
        ).fetchall()
        for metric, metric_id in ids.items()
    }
    print("\nwindow means (every sample, zeros included)")
    for metric, rows in samples.items():
        mean = statistics.fmean(v for _, v in rows) if rows else float("nan")
        print(f"  {metric:28s}{mean:8.2f}%   n={len(rows)}")

    thread_of = dict(
        db.execute(
            "select correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start < ? and end > ?",
            (t1, t0 - 60_000_000_000),
        )
    )
    thread_names = {}
    try:
        thread_names = {
            tid: strings.get(name_id, "")
            for tid, name_id in db.execute("select globalTid, nameId from ThreadNames")
        }
    except sqlite3.OperationalError:
        pass
    per_thread: dict = collections.defaultdict(lambda: [0, 0])
    for start, end, _, _, corr, _ in kernels:
        entry = per_thread[thread_of.get(corr)]
        entry[0] += 1
        entry[1] += min(end, t1) - max(start, t0)
    print("\ndevice time by launching thread (sum of kernel durations)")
    for tid, (count, total) in sorted(per_thread.items(), key=lambda item: -item[1][1])[
        :12
    ]:
        label = (
            f"{tid & 0xFFFFFF if tid is not None else None} {thread_names.get(tid, '')}"
        )
        print(
            f"  {label:40s}{count:>10d} kernels{total / 1e6:>11.1f} ms{100 * total / (t1 - t0):>7.1f}% of window"
        )

    busy = collections.defaultdict(lambda: [0, 0.0])
    for start, end, _, name_id, *_ in kernels:
        entry = busy[strings.get(name_id, str(name_id))]
        entry[0] += 1
        entry[1] += (end - start) / 1e6
    starts = [k[0] for k in kernels]
    max_len = max(k[1] - k[0] for k in kernels)
    per_kernel = collections.defaultdict(lambda: collections.defaultdict(list))
    for metric, rows in samples.items():
        for ts, value in rows:
            index = bisect.bisect_right(starts, ts) - 1
            names = set()
            while index >= 0 and kernels[index][0] >= ts - max_len:
                if kernels[index][0] <= ts <= kernels[index][1]:
                    names.add(strings.get(kernels[index][3], str(kernels[index][3])))
                index -= 1
            for name in names or {"<no kernel>"}:
                per_kernel[name][metric].append(value)
    header = " ".join(f"{m[:12]:>12}" for m in samples)
    print(f"\n{'kernel':90s} {'calls':>8} {'ms':>10} {header}")
    ranked = sorted(busy.items(), key=lambda item: -item[1][1])[: args.top]
    for name, (calls, ms) in ranked + [("<no kernel>", (0, 0.0))]:
        cells = []
        for metric in samples:
            values = per_kernel[name].get(metric)
            cells.append(
                f"{statistics.fmean(values):12.1f}" if values else f"{'-':>12}"
            )
        print(f"{name[:90]:90s} {calls:>8} {ms:>10.2f} " + " ".join(cells))


if __name__ == "__main__":
    main()
