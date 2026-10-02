# SPDX-License-Identifier: Apache-2.0
"""Vocoder step ledger of one serving boot with the prefix_count probe (COUNT=1 in
run_census_boot.sh), cut into each cell's timed window.

Per cell and step plan: steps, rows per step, step wall time, the vocoder's busy share of
the window, ready streams left behind by the step (ready before the step minus the rows it
took), queue wait of the rows, and a least squares fit of step time against rows
(ms = fixed + per_row * rows). For causal steps on a prefix cache tree, the share of rows
that ran cached, fell back to the whole history hop, and the mixed steps.

usage: python step_ledger.py <run dir>   (serve.log and <cell>.log, <cell>/measured)
"""

from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

STAMP = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d{3})"
STEP = re.compile(
    r"voc\.step t=([\d.]+) plan=(\S+) rows=(\d+) ms=([\d.]+) wait_max=([\d.]+) "
    r"wait_mean=([\d.]+) ready=(\d+)"
)
PREFIX = re.compile(
    r"prefix\.step t=([\d.]+) rows=(\d+) cached=(\d+) plain=(\d+) dropped=(\d+) "
    r"free_frames=(-?\d+) host_ms=([\d.]+)"
)


def window(bench_log: Path) -> tuple[float, float]:
    start = end = None
    for line in bench_log.read_text(errors="replace").splitlines():
        begun = re.search(STAMP + r".*Benchmarking \d+ requests", line)
        saved = re.search(STAMP + r".*Results saved to", line)
        if begun:
            start = datetime.fromisoformat(f"{begun.group(1)}.{begun.group(2)}")
        if saved and end is None and start is not None:
            end = datetime.fromisoformat(f"{saved.group(1)}.{saved.group(2)}")
    return start.timestamp(), end.timestamp()


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def fit(rows: list[int], ms: list[float]) -> tuple[float, float]:
    if len(set(rows)) < 2:
        return statistics.fmean(ms), 0.0
    slope, intercept = statistics.linear_regression(rows, ms)
    return intercept, slope


def main() -> None:
    run = Path(sys.argv[1])
    steps, prefix = [], []
    for line in (run / "serve.log").read_text(errors="replace").splitlines():
        if match := STEP.search(line):
            t, plan, rows, ms, wait_max, wait_mean, ready = match.groups()
            steps.append(
                (
                    float(t),
                    plan,
                    int(rows),
                    float(ms),
                    float(wait_max),
                    float(wait_mean),
                    int(ready),
                )
            )
        elif match := PREFIX.search(line):
            t, rows, cached, plain, dropped, free, host_ms = match.groups()
            prefix.append(
                (float(t), int(rows), int(cached), int(plain), int(dropped), int(free))
            )
    for bench_log in sorted(run.glob("*.log")):
        speed = run / bench_log.stem / "measured" / "speed_results.json"
        if not speed.exists():
            continue
        start, end = window(bench_log)
        span = end - start
        summary = json.loads(speed.read_text())["summary"]
        print(
            f"== {bench_log.stem}  window {span:.1f} s  req/s {summary['throughput_qps']}  "
            f"latency mean {summary['latency_mean_s']} s  first audio mean "
            f"{summary.get('audio_ttfp_mean_s')} s  audio mean {summary['audio_duration_mean_s']} s"
        )
        # a step's stamp is its end; keep the steps that ran inside the window
        cell = [
            step
            for step in steps
            if start <= step[0] - step[3] / 1e3 and step[0] <= end
        ]
        busy = sum(step[3] for step in cell) / 1e3
        print(
            f"   steps {len(cell)}  vocoder busy {busy:.1f} s = {100 * busy / span:.1f} % of the window"
        )
        by_plan = collections.defaultdict(list)
        for step in cell:
            by_plan[step[1]].append(step)
        for plan, group in sorted(by_plan.items()):
            rows = [step[2] for step in group]
            ms = [step[3] for step in group]
            left = [step[6] - step[2] for step in group]
            waits = [step[5] for step in group]
            fixed, per_row = fit(rows, ms)
            print(
                f"   {plan:14s} n {len(group):5d}  rows mean {statistics.fmean(rows):5.2f} "
                f"p50 {quantile(rows, 0.5):2d} p90 {quantile(rows, 0.9):2d}  "
                f"ms mean {statistics.fmean(ms):6.1f} p50 {quantile(ms, 0.5):6.1f} "
                f"p90 {quantile(ms, 0.9):6.1f}  busy {sum(ms) / 1e3:6.1f} s  "
                f"ms per row {sum(ms) / sum(rows):5.1f}  fit {fixed:5.1f} + {per_row:4.1f} x rows  "
                f"ready left mean {statistics.fmean(left):4.2f} (steps leaving any "
                f"{100 * sum(x > 0 for x in left) / len(left):4.1f} %)  "
                f"row wait mean {statistics.fmean(waits):6.1f} ms"
            )
            histogram = collections.Counter(rows)
            print(
                "      rows histogram "
                + " ".join(f"{size}:{histogram[size]}" for size in sorted(histogram))
            )
        cached = [p for p in prefix if start <= p[0] <= end]
        if cached:
            rows = sum(p[1] for p in cached)
            print(
                f"   prefix hops {len(cached)}  rows cached {100 * sum(p[2] for p in cached) / rows:.1f} %  "
                f"whole history {100 * sum(p[3] for p in cached) / rows:.1f} %  "
                f"mixed steps {100 * sum(p[2] > 0 and p[3] > 0 for p in cached) / len(cached):.1f} %  "
                f"least free frames {min(p[5] for p in cached)}"
            )
        gaps = [
            later[0] - later[3] / 1e3 - earlier[0]
            for earlier, later in zip(cell, cell[1:])
        ]
        if gaps:
            print(
                f"   idle between steps: total {sum(gaps):.1f} s, mean {1e3 * statistics.fmean(gaps):.1f} ms, "
                f"p90 {1e3 * quantile(gaps, 0.9):.1f} ms"
            )


if __name__ == "__main__":
    main()
