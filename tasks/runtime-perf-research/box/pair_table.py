"""Speech pair as PR table rows: each boot's mean of its two clean passes, the arms' boot values and the change of
their means; then every pass's QPS, so a pass that ran on or failed shows before the table is read.

usage: python3 pair_table.py <root> <base boots a1,a2> <branch boots b1,b2> [loads c64,c16]
"""

import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
base, branch = sys.argv[2].split(","), sys.argv[3].split(",")
loads = sys.argv[4].split(",") if len(sys.argv) > 4 else ["c64", "c16"]
ROWS = [("QPS", "throughput_qps", 1, 2), ("audio s/s", "audio_throughput_s_per_s", 1, 1), ("RTF mean", "rtf_mean", 1, 3),
        ("latency median (ms)", "latency_median_s", 1e3, 0), ("latency p95 (ms)", "latency_p95_s", 1e3, 0),
        ("latency p99 (ms)", "latency_p99_s", 1e3, 0), ("first audio mean (ms)", "audio_ttfp_mean_s", 1e3, 0),
        ("first audio p99 (ms)", "audio_ttfp_p99_s", 1e3, 0), ("inter chunk mean (ms)", "inter_chunk_mean_s", 1e3, 0),
        ("inter chunk p99 (ms)", "inter_chunk_p99_s", 1e3, 0)]


def summary(boot, load, rep):
    return json.loads((root / boot / f"{load}_clean/bench/{rep}/speed_results.json").read_text())["summary"]


def value(boot, load, key):
    return statistics.mean(summary(boot, load, rep)[key] for rep in ("rep0", "rep1"))


for load in loads:
    print(f"| {load} streamed | base | branch | change |")
    print("|---|---|---|---|")
    for label, key, scale, digits in ROWS:
        a = [value(boot, load, key) * scale for boot in base]
        b = [value(boot, load, key) * scale for boot in branch]
        print(f"| {label} | {' / '.join(f'{v:.{digits}f}' for v in a)} | {' / '.join(f'{v:.{digits}f}' for v in b)} "
              f"| {(statistics.mean(b) / statistics.mean(a) - 1) * 100:+.1f} % |")
    for boot in base + branch:
        passes = [summary(boot, load, rep) for rep in ("rep0", "rep1")]
        print(f"  {boot} {load} passes qps {' '.join(f'{s['throughput_qps']:.2f}' for s in passes)} failed "
              f"{sum(s['failed_requests'] for s in passes)} audio mean "
              f"{' '.join(f'{s['audio_duration_mean_s']:.2f}' for s in passes)}")
    print()
