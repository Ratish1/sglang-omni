"""GPU metrics per pipeline stage from one nsys profile with the probe, the CUDA trace and
GPU metrics sampling (run on the box).

Contexts of different processes time-slice the card, so a metric sample taken while a
kernel runs belongs to the process that launched that kernel. Each sample is attributed to
the stage of the kernel(s) executing at its timestamp (stage names from the probe's
"proc stage=" marks), or to "no kernel". Per stage it prints the share of samples, the mean
of every metric over that stage's samples, and the stage's kernel time; then the same per
kernel name inside each stage for the top kernels. A sample is the mean of its period, so a
stage's means are exact only for slices long against the period; samples that straddle a
switch mix two stages and are counted for each.

usage: python nsys_stage_metrics.py REPORT.sqlite [--bench-log bench.log] [--top 8]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import sqlite3
import statistics

from nsys_metrics import METRICS, bench_window

MARK = 34


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def stages_by_pid(db: sqlite3.Connection, strings: dict[int, str]) -> dict[int, str]:
    stages: dict[int, list[str]] = collections.defaultdict(list)
    for tid, text, text_id in db.execute(
        "select globalTid, text, textId from NVTX_EVENTS where eventType = ?", (MARK,)
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label and label.startswith("proc stage="):
            stage = label.split("=", 1)[1]
            if stage not in stages[pid_of(tid)]:
                stages[pid_of(tid)].append(stage)
    return {pid: "+".join(sorted(names)) for pid, names in stages.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    if args.bench_log:
        t0, t1 = bench_window(db, args.bench_log)
    else:
        t0, t1 = db.execute(
            "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
    stage_of = stages_by_pid(db, strings)
    kernels = db.execute(
        "select start, end, deviceId, demangledName, globalPid from CUPTI_ACTIVITY_KIND_KERNEL "
        "where end > ? and start < ? order by start",
        (t0, t1),
    ).fetchall()
    if not kernels:
        print("no kernel activity in the window")
        return
    device = collections.Counter(k[2] for k in kernels).most_common(1)[0][0]
    kernels = [k for k in kernels if k[2] == device]
    window_ms = (t1 - t0) / 1e6
    print(f"window {window_ms / 1e3:.3f} s, device {device}, {len(kernels)} kernels")

    def stage_name(global_pid: int) -> str:
        return stage_of.get(pid_of(global_pid), f"pid{pid_of(global_pid)}")

    kernel_ms: dict[str, float] = collections.defaultdict(float)
    kernel_ms_by_name: dict[tuple[str, str], list[float]] = collections.defaultdict(
        lambda: [0, 0.0]
    )
    for start, end, _, name_id, global_pid in kernels:
        duration_ms = (min(end, t1) - max(start, t0)) / 1e6
        stage = stage_name(global_pid)
        kernel_ms[stage] += duration_ms
        entry = kernel_ms_by_name[(stage, strings.get(name_id, str(name_id)))]
        entry[0] += 1
        entry[1] += duration_ms

    metric_ids = {}
    for metric_id, name in db.execute(
        "select metricId, metricName from TARGET_INFO_GPU_METRICS"
    ):
        for metric in METRICS:
            if str(name).startswith(metric):
                metric_ids[metric] = metric_id
    starts = [k[0] for k in kernels]
    longest = max(k[1] - k[0] for k in kernels)

    def running_at(timestamp: int) -> set[tuple[str, str]]:
        found = set()
        index = bisect.bisect_right(starts, timestamp) - 1
        while index >= 0 and kernels[index][0] >= timestamp - longest:
            start, end, _, name_id, global_pid = kernels[index]
            if start <= timestamp <= end:
                found.add((stage_name(global_pid), strings.get(name_id, str(name_id))))
            index -= 1
        return found

    by_stage: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    by_kernel: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    sample_count: dict[str, collections.Counter] = {}
    for metric, metric_id in metric_ids.items():
        counter: collections.Counter = collections.Counter()
        for timestamp, value in db.execute(
            "select timestamp, value from GPU_METRICS where metricId = ? and timestamp between ? and ?",
            (metric_id, t0, t1),
        ):
            running = running_at(timestamp)
            stages = {stage for stage, _ in running} or {"no kernel"}
            for stage in stages:
                by_stage[stage][metric].append(value)
                counter[stage] += 1
            for key in running:
                by_kernel[key][metric].append(value)
        sample_count[metric] = counter

    total_samples = sum(sample_count["SM Issue"].values()) or 1
    header = " ".join(f"{metric[:13]:>13}" for metric in metric_ids)
    print(
        f"\n{'stage':34s} {'samples %':>9} {'kernel ms':>10} {'kernel %':>8} {header}"
    )
    for stage, metrics in sorted(
        by_stage.items(), key=lambda item: -len(item[1]["SM Issue"])
    ):
        share = 100 * len(metrics["SM Issue"]) / total_samples
        cells = " ".join(
            f"{statistics.fmean(metrics[m]):13.2f}" if metrics[m] else f"{'-':>13}"
            for m in metric_ids
        )
        print(
            f"{stage[:34]:34s} {share:9.1f} {kernel_ms.get(stage, 0.0):10.1f} "
            f"{100 * kernel_ms.get(stage, 0.0) / window_ms:8.1f} {cells}"
        )

    for stage in sorted(kernel_ms, key=lambda name: -kernel_ms[name]):
        ranked = sorted(
            (
                (name, calls_ms)
                for (owner, name), calls_ms in kernel_ms_by_name.items()
                if owner == stage
            ),
            key=lambda item: -item[1][1],
        )[: args.top]
        print(f"\n{stage}: top kernels by device time")
        print(f"  {'kernel':80s} {'calls':>8} {'ms':>9} {header}")
        for name, (calls, ms) in ranked:
            metrics = by_kernel.get((stage, name), {})
            cells = " ".join(
                (
                    f"{statistics.fmean(metrics[m]):13.2f}"
                    if metrics.get(m)
                    else f"{'-':>13}"
                )
                for m in metric_ids
            )
            print(f"  {name[:80]:80s} {calls:>8} {ms:>9.1f} {cells}")


if __name__ == "__main__":
    main()
