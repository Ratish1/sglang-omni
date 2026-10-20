"""GPU metrics by what the device runs, from an nsys export with the gh100 GPU metrics and pipeline_census's owners.

Every metric sample inside the benchmark window is assigned by its timestamp: idle (no device interval of any owner),
one owner alone, or shared (two or more owners at once). For each bucket: its share of the window and the mean SM
Issue, SMs Active, Tensor Active, DRAM read and GR Active of its samples.

usage: python metrics_by_owner.py REPORT.sqlite --bench-log bench.log
"""

from __future__ import annotations

import argparse
import collections

from pipeline_census import Report

METRICS = (
    "GR Active [Throughput %]",
    "SMs Active [Throughput %]",
    "SM Issue [Throughput %]",
    "Tensor Active [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log", required=True)
    arguments = parser.parse_args()
    report = Report(arguments.report, arguments.bench_log)
    names = dict(report.db.execute("select metricName, metricId from TARGET_INFO_GPU_METRICS"))
    ids = {names[name]: name for name in METRICS}
    samples: dict[int, dict[str, float]] = collections.defaultdict(dict)
    for timestamp, metric_id, value in report.db.execute(
        f"select timestamp, metricId, value from GPU_METRICS where timestamp between ? and ? "
        f"and metricId in ({','.join('?' * len(ids))})",
        (report.t0, report.t1, *ids),
    ):
        samples[timestamp][ids[metric_id]] = value
    events = []
    for start, end, owner, *_rest in report.device:
        if end > report.t0 and start < report.t1:
            events.append((max(start, report.t0), 1, owner))
            events.append((min(end, report.t1), 0, owner))
        else:
            pass
    events.sort()
    active: collections.Counter[str] = collections.Counter()
    buckets: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    index = 0
    for timestamp in sorted(samples):
        while index < len(events) and events[index][0] <= timestamp:
            _, is_start, owner = events[index]
            if is_start:
                active[owner] += 1
            else:
                active[owner] -= 1
                if active[owner] <= 0:
                    del active[owner]
                else:
                    pass
            index += 1
        if not active:
            bucket = "idle"
        elif len(active) == 1:
            bucket = f"alone: {next(iter(active))}"
        else:
            bucket = "shared"
        buckets[bucket].append(samples[timestamp])
    total = sum(len(rows) for rows in buckets.values())
    short = ("GR", "SMs Active", "SM Issue", "Tensor", "DRAM read")
    print(f"{'bucket':<26} {'share':>7} " + " ".join(f"{label:>10}" for label in short))
    for bucket, rows in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        means = [sum(row.get(name, 0.0) for row in rows) / len(rows) for name in METRICS]
        print(
            f"{bucket:<26} {100 * len(rows) / total:6.1f}% " + " ".join(f"{mean:10.2f}" for mean in means)
        )


if __name__ == "__main__":
    main()
