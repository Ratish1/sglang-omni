#!/usr/bin/env python3
"""Average nsys GPU metrics over the timed SeedTTS cohort only.

The headline bench.log must contain:
  Warmup (16 requests)...
  Benchmarking 200 requests ...
  Results saved to ...

The SM window is [Benchmarking, Results saved]. Probe, preload, and warmup
are excluded. The window still holds the cohort's ramp (the first requests
filling the concurrency) and its drain (the last ones finishing), so quote
the window with the numbers.

Metrics are looked up by name in TARGET_INFO_GPU_METRICS: the ids depend on
the --gpu-metrics-set and the driver. The log's asctime is the host's local
clock, so the window is placed with the session's localTime, not utcTime.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime
from pathlib import Path

METRIC_NAMES = ("GR Active", "SMs Active", "SM Issue", "Tensor Active")

BENCH_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)\s+" r".*Benchmarking (\d+) requests"
)
SAVED_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)\s+.*Results saved to"
)


def _parse_log_ts(date_s: str, ms: str) -> datetime:
    return datetime.fromisoformat(f"{date_s.replace(' ', 'T')}.{ms}")


def parse_headline_window(bench_log: Path) -> tuple[datetime, datetime, int]:
    start: datetime | None = None
    end: datetime | None = None
    n_requests = 0
    for line in bench_log.read_text().splitlines():
        bench = BENCH_RE.match(line)
        if bench:
            start = _parse_log_ts(bench.group(1), bench.group(2))
            n_requests = int(bench.group(3))
            continue
        saved = SAVED_RE.match(line)
        if saved and start is not None:
            end = _parse_log_ts(saved.group(1), saved.group(2))
            break
    if start is None or end is None:
        raise ValueError(f"could not find Benchmarking/Results saved in {bench_log}")
    return start, end, n_requests


def session_start(con: sqlite3.Connection) -> datetime:
    row = con.execute("SELECT localTime FROM TARGET_INFO_SESSION_START_TIME").fetchone()
    if row is None:
        raise ValueError("no TARGET_INFO_SESSION_START_TIME in the export")
    return datetime.fromisoformat(row[0])


def metric_ids(con: sqlite3.Connection) -> dict[str, int]:
    ids: dict[str, int] = {}
    for metric_id, name in con.execute(
        "SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"
    ):
        for wanted in METRIC_NAMES:
            if name.startswith(wanted):
                ids[wanted] = metric_id
    missing = [name for name in METRIC_NAMES if name not in ids]
    if missing:
        raise ValueError(f"metrics not in TARGET_INFO_GPU_METRICS: {missing}")
    return ids


def metric_mean(
    con: sqlite3.Connection, metric_id: int, t0_ns: int, t1_ns: int
) -> tuple[float, int]:
    mean, n = con.execute(
        "SELECT AVG(value), COUNT(*) FROM GPU_METRICS "
        "WHERE metricId=? AND timestamp>=? AND timestamp<=?",
        (metric_id, t0_ns, t1_ns),
    ).fetchone()
    if not n:
        raise ValueError(f"no GPU_METRICS samples for metricId={metric_id}")
    return mean, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--bench-log", type=Path, required=True)
    args = parser.parse_args()

    con = sqlite3.connect(args.sqlite)
    bench0, bench1, n_requests = parse_headline_window(args.bench_log)
    sess = session_start(con)
    t0_ns = int((bench0 - sess).total_seconds() * 1e9)
    t1_ns = int((bench1 - sess).total_seconds() * 1e9)
    if t1_ns <= t0_ns:
        raise ValueError(f"empty window: session={sess} bench=[{bench0}, {bench1}]")

    print(f"session_start     {sess.isoformat()}  (host local clock)")
    print(f"bench_start       {bench0.isoformat()}  (after warmup)")
    print(f"bench_end         {bench1.isoformat()}")
    print(f"timed_requests    {n_requests}")
    print(f"session_rel_s     {t0_ns / 1e9:.3f} .. {t1_ns / 1e9:.3f}")
    print(f"window_s          {(t1_ns - t0_ns) / 1e9:.3f}")
    for name, metric_id in metric_ids(con).items():
        mean, n = metric_mean(con, metric_id, t0_ns, t1_ns)
        print(f"{name:16s}  {mean:.4f}%  n={n}  (metricId {metric_id})")
    con.close()


if __name__ == "__main__":
    main()
