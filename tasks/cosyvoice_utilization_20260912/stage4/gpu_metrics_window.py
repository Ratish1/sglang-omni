"""Window means of the nsys GPU metrics (gh100 set) over a benchmark's timed requests.

usage: python gpu_metrics_window.py REPORT.sqlite --bench-log bench.log
"""

from __future__ import annotations

import argparse
import sqlite3

from nsys_metrics import bench_window

METRICS = (
    "GR Active [Throughput %]",
    "SMs Active [Throughput %]",
    "SM Issue [Throughput %]",
    "Tensor Active [Throughput %]",
    "Compute Warps in Flight [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log", required=True)
    arguments = parser.parse_args()
    db = sqlite3.connect(arguments.report)
    tables = {
        row[0]
        for row in db.execute("select name from sqlite_master where type='table'")
    }
    if "GPU_METRICS" not in tables:
        print("no GPU_METRICS table in this report")
        return
    else:
        pass
    start_ns, end_ns = bench_window(db, arguments.bench_log)
    names = dict(db.execute("select metricName, metricId from TARGET_INFO_GPU_METRICS"))
    print(f"window {(end_ns - start_ns) / 1e9:.1f} s")
    for name in METRICS:
        metric_id = names.get(name)
        if metric_id is None:
            print(f"  {name}: absent")
            continue
        else:
            pass
        mean, samples = db.execute(
            "select avg(value), count(*) from GPU_METRICS where metricId = ? and timestamp between ? and ?",
            (metric_id, start_ns, end_ns),
        ).fetchone()
        print(f"  {name:<44} mean {mean or 0:6.2f}  samples {samples}")


if __name__ == "__main__":
    main()
