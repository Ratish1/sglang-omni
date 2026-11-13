"""A PR body table: each arm's boot values and the change of the arm means, from every measured pass's speed_results.json.

usage: python3 body_table.py <run dir> <cell, e.g. c16> "<title>" main=boot,boot pr=boot,boot

A boot's value is the mean over its clean repeats and its events pass; the warm pass is skipped.
"""

import json
import statistics
import sys
from pathlib import Path

ROWS = [
    ("QPS", "throughput_qps", 1, "{:.2f}"),
    ("RTF mean", "rtf_mean", 1, "{:.3f}"),
    ("latency median (ms)", "latency_median_s", 1000, "{:.0f}"),
    ("latency p99 (ms)", "latency_p99_s", 1000, "{:.0f}"),
    ("first audio mean (ms)", "audio_ttfp_mean_s", 1000, "{:.0f}"),
    ("first audio p99 (ms)", "audio_ttfp_p99_s", 1000, "{:.0f}"),
    ("inter chunk mean (ms)", "inter_chunk_mean_s", 1000, "{:.0f}"),
    ("inter chunk p99 (ms)", "inter_chunk_p99_s", 1000, "{:.0f}"),
]


def boot_values(boot: Path, cell: str) -> dict[str, float]:
    summaries = []
    for directory in sorted(boot.glob(f"{cell}_*")):
        if directory.name.endswith("_warm"):
            continue
        else:
            pass
        for path in sorted(directory.glob("bench/**/speed_results.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
            if summary.get("failed_requests", 0) == 0:
                summaries.append(summary)
            else:
                pass
    return {
        key: statistics.fmean(summary[key] for summary in summaries)
        for _, key, _, _ in ROWS
    }


def main() -> None:
    run_dir, cell, title = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    arms = []
    for spec in sys.argv[4:]:
        arm, boots = spec.split("=")
        arms.append(
            (arm, [boot_values(run_dir / boot, cell) for boot in boots.split(",")])
        )
    (base_name, base), (pr_name, pr) = arms
    print(f"| {title} | {base_name} | {pr_name} | change |")
    print("|---|---|---|---|")
    for label, key, scale, form in ROWS:
        base_values = [boot[key] * scale for boot in base]
        pr_values = [boot[key] * scale for boot in pr]
        change = (statistics.fmean(pr_values) / statistics.fmean(base_values) - 1) * 100
        print(
            f"| {label} | {' / '.join(form.format(value) for value in base_values)} | "
            f"{' / '.join(form.format(value) for value in pr_values)} | {change:+.1f} % |"
        )


if __name__ == "__main__":
    main()
