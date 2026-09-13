"""GPU metrics means from an nsys SQLite export made with --gpu-metrics-devices.

Usage: python nsys_gpu_metrics.py <trace.sqlite> [<trace.sqlite> ...]

Prints, per file, the mean over the sampled window of GR Active, SMs Active, Compute
Warps in Flight (throughput percent and warps per cycle) and the GPC clock, with the
sample count and the window length. The means are over every sample of the export, so
start the capture only when traffic is steady.
"""

import sqlite3
import sys

WANTED = (
    "GR Active",
    "SMs Active",
    "Compute Warps in Flight",
    "GPC Clock Frequency",
)


def report(path: str) -> None:
    con = sqlite3.connect(path)
    names = dict(
        con.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")
    )
    print(f"== {path}")
    for metric_id, name in sorted(names.items(), key=lambda kv: kv[1]):
        if not any(name.startswith(w) for w in WANTED):
            continue
        mean, count, first, last = con.execute(
            "select avg(value), count(*), min(timestamp), max(timestamp) "
            "from GPU_METRICS where metricId = ?",
            (metric_id,),
        ).fetchone()
        if count == 0:
            continue
        span_s = (last - first) / 1e9
        print(
            f"  {name:45s} mean {mean:12.1f}  samples {count:7d}  window {span_s:5.1f} s"
        )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path in sys.argv[1:]:
        report(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
