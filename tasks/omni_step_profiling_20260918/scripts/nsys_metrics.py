"""GPU metrics per kernel from an nsys profile sqlite export (run on the box).

Takes the kernel executions (CUPTI_ACTIVITY_KIND_KERNEL, graph nodes included) and the
GPU metric samples (GPU_METRICS, ad10x set) of one device over the collection window.
Prints the window totals (kernel busy time, metric means), then attributes every metric
sample to the kernel executing at its timestamp (kernels on the same device may overlap
across streams; a sample counts for each) and prints, per kernel short name ranked by
device time: calls, total ms, and the mean of SM Issue, SMs Active, Tensor Active, warps
in flight and DRAM read / write bandwidth over its samples.

usage: python nsys_metrics.py REPORT.sqlite [--top 30]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import sqlite3
import statistics

METRICS = (
    "SM Issue [Throughput %]",
    "SMs Active [Throughput %]",
    "Tensor Active [Throughput %]",
    "Compute Warps in Flight [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    kernels = db.execute(
        "select start, end, deviceId, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    if not kernels:
        print("no kernel activity in the report")
        return
    device = collections.Counter(k[2] for k in kernels).most_common(1)[0][0]
    kernels = [k for k in kernels if k[2] == device]
    t0, t1 = kernels[0][0], max(k[1] for k in kernels)
    busy = collections.defaultdict(lambda: [0, 0.0])
    for start, end, _, name_id in kernels:
        entry = busy[strings.get(name_id, str(name_id))]
        entry[0] += 1
        entry[1] += (end - start) / 1e6
    print(
        f"device {device}: {len(kernels)} kernels over {(t1 - t0) / 1e6:.1f} ms, busy sum {sum(v[1] for v in busy.values()):.1f} ms"
    )

    ids = dict(
        (name, metric_id)
        for metric_id, name in db.execute(
            "select metricId, metricName from TARGET_INFO_GPU_METRICS"
        )
    )
    starts = [k[0] for k in kernels]
    order_by_end = sorted(range(len(kernels)), key=lambda i: kernels[i][1])
    max_len = max(k[1] - k[0] for k in kernels)
    per_kernel = collections.defaultdict(lambda: collections.defaultdict(list))
    window = {}
    for metric in METRICS:
        if metric not in ids:
            continue
        samples = db.execute(
            "select timestamp, value from GPU_METRICS where metricId = ? and timestamp between ? and ?",
            (ids[metric], t0, t1),
        ).fetchall()
        window[metric] = (
            statistics.fmean(v for _, v in samples) if samples else float("nan")
        )
        for ts, value in samples:
            index = bisect.bisect_right(starts, ts) - 1
            names = set()
            while index >= 0 and kernels[index][0] >= ts - max_len:
                start, end, _, name_id = kernels[index]
                if start <= ts <= end:
                    names.add(strings.get(name_id, str(name_id)))
                index -= 1
            for name in names or {"<idle>"}:
                per_kernel[name][metric].append(value)
    del order_by_end
    print(
        "window means: "
        + ", ".join(f"{m.split(' [')[0]} {v:.1f}" for m, v in window.items())
    )
    header = " ".join(f"{m.split(' [')[0][:12]:>12}" for m in METRICS)
    print(f"\n{'kernel':60s} {'calls':>7} {'ms':>9} {header}")
    ranked = sorted(busy.items(), key=lambda item: -item[1][1])[: args.top]
    for name, (calls, ms) in ranked + [("<idle>", (0, 0.0))]:
        cells = []
        for metric in METRICS:
            values = per_kernel[name].get(metric)
            cells.append(
                f"{statistics.fmean(values):12.1f}" if values else f"{'-':>12}"
            )
        print(f"{name[:60]:60s} {calls:>7} {ms:>9.2f} " + " ".join(cells))


if __name__ == "__main__":
    main()
