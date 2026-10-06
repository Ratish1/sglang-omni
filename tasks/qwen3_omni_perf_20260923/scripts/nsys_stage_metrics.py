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

usage: python nsys_stage_metrics.py REPORT.sqlite [--window window.txt] [--top 8]
"""

from __future__ import annotations

import argparse
import collections
import heapq
import sqlite3
import statistics

from nsys_metrics import METRICS
from nsys_stage_ledger import pid_of, session_window

MARK = 34


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
    parser.add_argument("--window")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
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
    metric_of_id = {metric_id: metric for metric, metric_id in metric_ids.items()}
    by_stage: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    by_kernel: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    # one sweep in time order: kernels enter the running set at their start and leave it
    # once a sample is past their end
    running_ends: list[tuple[int, int]] = []
    next_kernel = 0
    sampled_at = None
    running: set[tuple[str, str]] = set()
    placeholders = ",".join("?" for _ in metric_of_id)
    for timestamp, metric_id, value in db.execute(
        f"select timestamp, metricId, value from GPU_METRICS where metricId in ({placeholders}) "
        "and timestamp between ? and ? order by timestamp",
        (*metric_of_id, t0, t1),
    ):
        if timestamp != sampled_at:
            sampled_at = timestamp
            while next_kernel < len(kernels) and kernels[next_kernel][0] <= timestamp:
                heapq.heappush(running_ends, (kernels[next_kernel][1], next_kernel))
                next_kernel += 1
            while running_ends and running_ends[0][0] < timestamp:
                heapq.heappop(running_ends)
            running = {
                (
                    stage_name(kernels[index][4]),
                    strings.get(kernels[index][3], str(kernels[index][3])),
                )
                for _, index in running_ends
            }
        metric = metric_of_id[metric_id]
        for stage in {stage for stage, _ in running} or {"no kernel"}:
            by_stage[stage][metric].append(value)
        for key in running:
            by_kernel[key][metric].append(value)

    total_samples = sum(len(metrics["SM Issue"]) for metrics in by_stage.values()) or 1
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
