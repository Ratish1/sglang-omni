"""PR body table of one A/B pair: both arms' values and the change per cell, with RTF, the
median, p95 and p99 rows, WER and speaker similarity.
usage: python3 pr_table.py <arm dir A> <name A> <arm dir B> <name B> <cell[:label]>..."""

import json
import sys
from pathlib import Path

SPEED_ROWS = (
    ("throughput_qps", "req/s", 3, 1),
    ("audio_throughput_s_per_s", "audio s/s", 2, 1),
    ("rtf_mean", "RTF mean", 4, 1),
    ("rtf_median", "RTF median", 4, 1),
    ("rtf_p95", "RTF p95", 4, 1),
    ("rtf_p99", "RTF p99", 4, 1),
    ("audio_ttfp_median_s", "first audio median (s)", 3, 1),
    ("audio_ttfp_p95_s", "first audio p95 (s)", 3, 1),
    ("audio_ttfp_p99_s", "first audio p99 (s)", 3, 1),
    ("inter_chunk_mean_s", "inter chunk mean (s)", 3, 1),
    ("inter_chunk_p95_s", "inter chunk p95 (s)", 3, 1),
    ("inter_chunk_p99_s", "inter chunk p99 (s)", 3, 1),
    ("latency_median_s", "latency median (s)", 3, 1),
    ("latency_p95_s", "latency p95 (s)", 3, 1),
    ("latency_p99_s", "latency p99 (s)", 3, 1),
)


def read(arm: Path, cell: str, name: str) -> dict:
    path = arm / cell / name
    return json.loads(path.read_text())["summary"] if path.exists() else {}


arm_a, name_a, arm_b, name_b = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
print(f"| cell | metric | {name_a} | {name_b} | change |")
print("| --- | --- | --- | --- | --- |")
for spec in sys.argv[5:]:
    cell, _, label = spec.partition(":")
    label = label or cell
    speed_a, speed_b = read(arm_a, cell, "speed_results.json"), read(arm_b, cell, "speed_results.json")
    for key, metric, digits, _ in SPEED_ROWS:
        a, b = speed_a.get(key), speed_b.get(key)
        if a is None or b is None:
            continue
        print(f"| {label} | {metric} | {a:.{digits}f} | {b:.{digits}f} | {100 * (b - a) / a:+.1f} % |")
    wer_a, wer_b = read(arm_a, cell, "wer_results.json"), read(arm_b, cell, "wer_results.json")
    if wer_a and wer_b:
        a, b = 100 * wer_a["wer_corpus"], 100 * wer_b["wer_corpus"]
        print(f"| {label} | WER (%) | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
    sim_a, sim_b = read(arm_a, cell, "similarity_results.json"), read(arm_b, cell, "similarity_results.json")
    if sim_a and sim_b:
        a, b = sim_a["speaker_similarity_mean"], sim_b["speaker_similarity_mean"]
        print(f"| {label} | SIM | {a:.2f} | {b:.2f} | {b - a:+.2f} |")
