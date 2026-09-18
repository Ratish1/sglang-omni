"""Read GPU metrics and CUDA graph node activity from an nsys sqlite export (on the box).

Prints the schema of CUDA_GRAPH_NODE_EVENTS and GPU_METRICS, then over the capture
window: per metric, mean and percentiles across samples; and the same metrics split by
which graphs are executing at each sample (graph id -> node count, so talker, predictor
and vocoder graphs are told apart by their node counts), from the node event intervals.

usage: python nsys_metrics.py REPORT.sqlite
"""

from __future__ import annotations

import bisect
import collections
import sqlite3
import statistics
import sys

METRICS = (
    "SMs Active [Throughput %]",
    "SM Issue [Throughput %]",
    "Tensor Active [Throughput %]",
    "Compute Warps in Flight [Throughput %]",
    "Unallocated Warps in Active SMs [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
    "DRAM Write Bandwidth [Throughput %]",
    "GR Active [Throughput %]",
)


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main() -> None:
    db = sqlite3.connect(sys.argv[1])
    for table in ("CUDA_GRAPH_NODE_EVENTS", "GPU_METRICS", "TARGET_INFO_GPU_METRICS"):
        cols = [row[1] for row in db.execute(f"pragma table_info({table})")]
        print(f"{table}: {cols}")
    print(
        "sample node events:",
        db.execute("select * from CUDA_GRAPH_NODE_EVENTS limit 3").fetchall(),
    )

    ids = {
        name: metric_id
        for metric_id, name in db.execute(
            "select metricId, metricName from TARGET_INFO_GPU_METRICS"
        )
    }
    node_cols = [
        row[1] for row in db.execute("pragma table_info(CUDA_GRAPH_NODE_EVENTS)")
    ]
    graph_col = "graphId" if "graphId" in node_cols else None
    intervals = []
    if graph_col and "start" in node_cols and "end" in node_cols:
        graph_sizes = collections.Counter(
            gid
            for (gid,) in db.execute(f"select {graph_col} from CUDA_GRAPH_NODE_EVENTS")
        )
        for start, end, gid in db.execute(
            f"select start, end, {graph_col} from CUDA_GRAPH_NODE_EVENTS order by start"
        ):
            intervals.append((start, end, gid))
        print("graphs by node events:", graph_sizes.most_common(20))
    t0, t1 = (
        (intervals[0][0], max(end for _, end, _ in intervals))
        if intervals
        else (None, None)
    )
    starts = [s for s, _, _ in intervals]

    print(f"\nwindow {((t1 - t0) / 1e6) if t0 else 'unknown'} ms")
    print(f"{'metric':46s} {'mean':>7} {'p10':>7} {'p50':>7} {'p90':>7} {'samples':>8}")
    for name in METRICS:
        if name not in ids:
            continue
        rows = db.execute(
            "select timestamp, value from GPU_METRICS where metricId = ?"
            + (" and timestamp between ? and ?" if t0 else ""),
            (ids[name], t0, t1) if t0 else (ids[name],),
        ).fetchall()
        values = [v for _, v in rows]
        if not values:
            continue
        print(
            f"{name:46s} {statistics.fmean(values):7.1f} {pct(values, .1):7.1f} "
            f"{pct(values, .5):7.1f} {pct(values, .9):7.1f} {len(values):>8}"
        )
        if not intervals:
            continue
        by_graph = collections.defaultdict(list)
        for ts, value in rows:
            index = bisect.bisect_right(starts, ts) - 1
            active = set()
            probe = index
            while probe >= 0 and len(active) < 3 and starts[probe] > ts - 5_000_000:
                start, end, gid = intervals[probe]
                if start <= ts <= end:
                    active.add(gid)
                probe -= 1
            key = tuple(sorted(active)) if active else ("idle",)
            by_graph[key].append(value)
        for key, vals in sorted(by_graph.items(), key=lambda item: -len(item[1]))[:8]:
            print(
                f"    graphs {str(key):40s} mean {statistics.fmean(vals):6.1f} samples {len(vals)}"
            )


if __name__ == "__main__":
    main()
