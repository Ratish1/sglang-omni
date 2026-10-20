"""Compact PR body table of one A/B pair, median / p95 / p99 in one row per metric.
usage: python3 pr_table_compact.py <arm dir A> <name A> <arm dir B> <name B> <cell[:label]>..."""

import json
import sys
from pathlib import Path

ROWS = (
    ("req/s", ("throughput_qps",), 3),
    ("RTF mean / p95 / p99", ("rtf_mean", "rtf_p95", "rtf_p99"), 4),
    ("first audio median / p95 / p99 (s)", ("audio_ttfp_median_s", "audio_ttfp_p95_s", "audio_ttfp_p99_s"), 3),
    ("inter chunk mean / p95 / p99 (s)", ("inter_chunk_mean_s", "inter_chunk_p95_s", "inter_chunk_p99_s"), 3),
    ("latency median / p95 / p99 (s)", ("latency_median_s", "latency_p95_s", "latency_p99_s"), 3),
)


def read(arm: Path, cell: str, name: str) -> dict:
    path = arm / cell / name
    return json.loads(path.read_text())["summary"] if path.exists() else {}


arm_a, name_a, arm_b, name_b = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
print(f"| cell | metric | {name_a} | {name_b} | delta |")
print("|---|---|---|---|---|")
for spec in sys.argv[5:]:
    cell, _, label = spec.partition(":")
    a, b = read(arm_a, cell, "speed_results.json"), read(arm_b, cell, "speed_results.json")
    for metric, keys, digits in ROWS:
        if any(a.get(key) is None or b.get(key) is None for key in keys):
            continue
        values_a = " / ".join(f"{a[key]:.{digits}f}" for key in keys)
        values_b = " / ".join(f"{b[key]:.{digits}f}" for key in keys)
        deltas = " / ".join(f"{100 * (b[key] - a[key]) / a[key]:+.1f}" for key in keys)
        print(f"| {label} | {metric} | {values_a} | {values_b} | {deltas} % |")
    wer_a, wer_b = read(arm_a, cell, "wer_results.json"), read(arm_b, cell, "wer_results.json")
    sim_a, sim_b = read(arm_a, cell, "similarity_results.json"), read(arm_b, cell, "similarity_results.json")
    if wer_a and wer_b and sim_a and sim_b:
        print(
            f"| {label} | WER (%) / SIM | {100 * wer_a['wer_corpus']:.2f} / {sim_a['speaker_similarity_mean']:.2f} "
            f"| {100 * wer_b['wer_corpus']:.2f} / {sim_b['speaker_similarity_mean']:.2f} | |"
        )
