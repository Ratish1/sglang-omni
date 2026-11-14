"""Run Nsight Systems' own stats reports and recipes on one cell report and print their top rows.

Usage: python3 summarize_cell.py <cell.nsys-rep> [--rows N]
Writes <cell>_<report>.csv and <cell>_<recipe>/ beside the report.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

STATS_REPORTS = (
    "nvtx_sum",
    "nvtx_gpu_proj_sum",
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_kern_sum:nvtx-name:base",
    "cuda_kern_exec_sum:base",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "osrt_sum",
)
RECIPES = ("cuda_api_sync", "cuda_memcpy_async", "cuda_memcpy_sync", "gpu_gaps", "gpu_time_util")
METRIC_RECIPE = "gpu_metric_util_sum"


def run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"## FAILED {' '.join(command[:3])}: {completed.stderr.strip()[-400:]}")
    else:
        pass


def print_csv(path: Path, rows: int) -> None:
    with path.open() as handle:
        table = list(csv.reader(handle))
    print(f"## {path.name} ({max(len(table) - 1, 0)} rows)")
    for row in table[: rows + 1]:
        print(" | ".join(cell[:90] for cell in row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--rows", type=int, default=25)
    args = parser.parse_args()
    stem = args.report.with_suffix("")
    run(
        [
            "nsys",
            "stats",
            "--force-export=true",
            "--format=csv",
            f"--output={stem}",
            *[f"--report={report}" for report in STATS_REPORTS],
            str(args.report),
        ]
    )
    for recipe in RECIPES:
        run(["nsys", "recipe", recipe, "--input", str(args.report), "--output", f"{stem}_{recipe}", "--force-overwrite"])
    run(["nsys", "recipe", METRIC_RECIPE, "--csv", "--input", str(args.report), "--output", f"{stem}_{METRIC_RECIPE}", "--force-overwrite"])
    for path in sorted(stem.parent.glob(f"{stem.name}_*.csv")):
        print_csv(path, args.rows)
    for path in sorted(stem.parent.glob(f"{stem.name}_*/*.csv")):
        print_csv(path, args.rows)


if __name__ == "__main__":
    sys.exit(main())
