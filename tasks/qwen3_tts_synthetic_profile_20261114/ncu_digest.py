"""Per-kernel Nsight Compute digest of one profiled step: time, speed of light, issue, occupancy, grid.

Usage: python3 ncu_digest.py <step.ncu-rep> [--rows N]
Rows are kernel names, time-weighted over their launches in the step.
"""

import argparse
import collections
import csv
import io
import subprocess
from pathlib import Path

DURATION = "gpu__time_duration.sum"
METRICS = {
    "dram %": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory %": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm %": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "issue %": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "occupancy %": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "waves": "launch__waves_per_multiprocessor",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
}


def number(text: str) -> float:
    return float(text.replace(",", "")) if text not in ("", "n/a") else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--rows", type=int, default=30)
    args = parser.parse_args()
    output = subprocess.run(
        [
            "ncu",
            "--import",
            str(args.report),
            "--csv",
            "--print-units",
            "base",
            "--page",
            "raw",
            "--metrics",
            ",".join([DURATION, *METRICS.values()]),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = list(csv.DictReader(io.StringIO(output)))[1:]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["Kernel Name"][:70]].append(row)
    step_time = sum(number(row[DURATION]) for row in rows)
    print(f"{len(rows)} kernels, {step_time / 1e3:.1f} us of kernel time in the step")
    print("share % | count | mean us | " + " | ".join(METRICS) + " | kernel")
    for name, launches in sorted(groups.items(), key=lambda item: -sum(number(r[DURATION]) for r in item[1]))[: args.rows]:
        durations = [number(r[DURATION]) for r in launches]
        total = sum(durations)
        weighted = [
            sum(number(r[metric]) * d for r, d in zip(launches, durations)) / max(total, 1.0)
            for metric in METRICS.values()
        ]
        print(
            f"{100 * total / step_time:6.1f} | {len(launches):5d} | {total / len(launches) / 1e3:7.2f} | "
            + " | ".join(f"{value:7.1f}" for value in weighted)
            + f" | {name}"
        )
    for label, metric in (("dram %", METRICS["dram %"]), ("issue %", METRICS["issue %"])):
        weighted = sum(number(r[metric]) * number(r[DURATION]) for r in rows) / max(step_time, 1.0)
        print(f"step time-weighted {label}: {weighted:.1f}")


if __name__ == "__main__":
    main()
