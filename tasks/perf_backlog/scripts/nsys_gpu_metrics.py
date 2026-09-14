"""GPU metrics means from an nsys SQLite export made with --gpu-metrics-devices.

Usage: python nsys_gpu_metrics.py <trace.sqlite> [--bench-log client.log] [--window T0 T1]

Prints the mean over the window of GR Active (any engine work), SMs Active (share of
SMs with a warp resident), SM Issue (share of issue slots used, the utilization rate),
Tensor Active, warps in flight and unallocated warps (occupancy), DRAM bandwidth and
the GPC clock, with the sample count, the window length and the share of samples at
zero (the idle the mean carries). Metrics are looked up by name in
TARGET_INFO_GPU_METRICS, never by id.

The window is the whole export unless one is given. --bench-log takes the benchmark
client log and uses its "Benchmarking N requests" and "Results saved to" lines, the
timed cohort, converted through the export's session start on the same host clock
(localTime, the clock the log's asctime uses). --window takes session relative seconds.
The cohort window includes the ramp and the drain of the cohort; both windows should
be reported with the numbers they produced.
"""

import argparse
import re
import sqlite3
from datetime import datetime

WANTED = (
    "GR Active",
    "SMs Active",
    "SM Issue",
    "Tensor Active",
    "Compute Warps in Flight",
    "Unallocated Warps in Active SMs",
    "DRAM Read Bandwidth",
    "DRAM Write Bandwidth",
    "GPC Clock Frequency",
)

LOG_TIME = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) "
BENCH_START = re.compile(LOG_TIME + r".*Benchmarking \d+ requests")
BENCH_END = re.compile(LOG_TIME + r".*Results saved to")


def cohort_window(con: sqlite3.Connection, bench_log: str) -> tuple[int, int]:
    """Session relative ns of the timed cohort, from the client log's own lines."""
    start = end = None
    for line in open(bench_log):
        if BENCH_START.match(line):
            start = BENCH_START.match(line).group(1)
        elif start is not None and BENCH_END.match(line):
            end = BENCH_END.match(line).group(1)
            break
    if start is None or end is None:
        raise SystemExit(f"{bench_log}: no Benchmarking / Results saved lines")
    (session_local,) = con.execute(
        "select localTime from TARGET_INFO_SESSION_START_TIME"
    ).fetchone()
    session = datetime.fromisoformat(session_local)
    to_ns = lambda text: int(
        (datetime.strptime(text, "%Y-%m-%d %H:%M:%S,%f") - session).total_seconds()
        * 1e9
    )
    return to_ns(start), to_ns(end)


def report(path: str, window: tuple[int, int] | None) -> None:
    con = sqlite3.connect(path)
    names = dict(
        con.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")
    )
    bounds = ""
    params: tuple = ()
    if window is not None:
        bounds = " and timestamp between ? and ?"
        params = window
        print(f"== {path}  window {window[0] / 1e9:.3f} .. {window[1] / 1e9:.3f} s")
    else:
        print(f"== {path}  whole export")
    for metric_id, name in sorted(names.items(), key=lambda kv: kv[1]):
        if not any(name.startswith(w) for w in WANTED):
            continue
        mean, count, first, last, zeros = con.execute(
            "select avg(value), count(*), min(timestamp), max(timestamp), "
            "sum(value = 0) from GPU_METRICS where metricId = ?" + bounds,
            (metric_id, *params),
        ).fetchone()
        if not count:
            continue
        span_s = (last - first) / 1e9
        if "Clock Frequency" in name:
            # note(ratish): nsys names the clock metric MHz but stores Hz.
            mean = mean / 1e6
        print(
            f"  {name:45s} mean {mean:12.1f}  samples {count:8d}  "
            f"span {span_s:6.1f} s  zero {100.0 * zeros / count:5.1f}%"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--bench-log", help="client log; window = its timed cohort")
    parser.add_argument(
        "--window", nargs=2, type=float, metavar=("T0", "T1"), help="session seconds"
    )
    args = parser.parse_args()
    window = None
    if args.bench_log:
        window = cohort_window(sqlite3.connect(args.sqlite), args.bench_log)
    elif args.window:
        window = (int(args.window[0] * 1e9), int(args.window[1] * 1e9))
    report(args.sqlite, window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
