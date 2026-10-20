"""Both arms' values and the change for the A/B cells of an ab_pair.sh run.
usage: python3 pair_table.py <arm dir A> <arm dir B> [cells...]"""

import json
import sys
from pathlib import Path

METRICS = (
    ("throughput_qps", "req/s", 3),
    ("audio_throughput_s_per_s", "audio s/s", 2),
    ("rtf_mean", "RTF mean", 4),
    ("audio_ttfp_mean_s", "first audio mean s", 3),
    ("audio_ttfp_p95_s", "first audio p95 s", 3),
    ("audio_ttfp_p99_s", "first audio p99 s", 3),
    ("inter_chunk_mean_s", "inter chunk mean s", 3),
    ("inter_chunk_p99_s", "inter chunk p99 s", 3),
    ("latency_mean_s", "latency mean s", 3),
    ("latency_p99_s", "latency p99 s", 3),
    ("failed_requests", "failed", 0),
)

arm_a, arm_b = Path(sys.argv[1]), Path(sys.argv[2])
cells = sys.argv[3:] or ["c1", "c16", "b16"]
print(f"| cell | metric | {arm_a.name} | {arm_b.name} | change |")
print("| --- | --- | --- | --- | --- |")
for cell in cells:
    a = json.loads((arm_a / cell / "speed_results.json").read_text())["summary"]
    b = json.loads((arm_b / cell / "speed_results.json").read_text())["summary"]
    for key, label, digits in METRICS:
        if a.get(key) is None or b.get(key) is None:
            continue
        change = "" if not a[key] else f"{(b[key] - a[key]) / a[key] * 100:+.1f} %"
        print(f"| {cell} | {label} | {a[key]:.{digits}f} | {b[key]:.{digits}f} | {change} |")
