"""Every metric row of one same time pair per cell, both arms and the change: req/s, RTF, first audio mean, p95 and
p99, inter chunk mean, p95 and p99, latency mean, p95 and p99, playback underrun, requests, token cap hits, WER below
50 % with the count above, and speaker similarity.
usage: python3 pr_rows_c20261012.py <arm dir A> <name A> <arm dir B> <name B> <cell>..."""

import json
import sys
from pathlib import Path

SPEED_ROWS = (
    ("throughput_qps", "req/s", 3),
    ("rtf_mean", "RTF", 4),
    ("audio_ttfp_mean_s", "first audio mean (s)", 3),
    ("audio_ttfp_p95_s", "first audio p95 (s)", 3),
    ("audio_ttfp_p99_s", "first audio p99 (s)", 3),
    ("inter_chunk_mean_s", "inter chunk mean (s)", 3),
    ("inter_chunk_p95_s", "inter chunk p95 (s)", 3),
    ("inter_chunk_p99_s", "inter chunk p99 (s)", 3),
    ("latency_mean_s", "latency mean (s)", 3),
    ("latency_p95_s", "latency p95 (s)", 3),
    ("latency_p99_s", "latency p99 (s)", 3),
    ("max_playback_underrun_mean_s", "underrun mean (s)", 3),
)
COUNT_ROWS = (
    ("completed_requests", "completed"),
    ("failed_requests", "failed"),
    ("max_token_hits", "at the token cap"),
)


def read(arm: Path, cell: str, name: str) -> dict:
    path = arm / cell / name
    return json.loads(path.read_text())["summary"] if path.exists() else {}


arm_a, name_a, arm_b, name_b = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
print(f"| cell | metric | {name_a} | {name_b} | change |")
print("| --- | --- | --- | --- | --- |")
for cell in sys.argv[5:]:
    speed_a, speed_b = read(arm_a, cell, "speed_results.json"), read(arm_b, cell, "speed_results.json")
    for key, metric, digits in SPEED_ROWS:
        a, b = speed_a.get(key), speed_b.get(key)
        if a is None or b is None:
            continue
        change = f"{100 * (b - a) / a:+.1f} %" if a else ""
        print(f"| {cell} | {metric} | {a:.{digits}f} | {b:.{digits}f} | {change} |")
    for key, metric in COUNT_ROWS:
        print(f"| {cell} | {metric} | {speed_a.get(key)} | {speed_b.get(key)} | |")
    wer_a, wer_b = read(arm_a, cell, "wer_results.json"), read(arm_b, cell, "wer_results.json")
    if wer_a and wer_b:
        print(
            f"| {cell} | WER below 50 % (above) | {100 * wer_a['wer_below_50_corpus']:.3f} % ({wer_a['n_above_50_pct_wer']}) "
            f"| {100 * wer_b['wer_below_50_corpus']:.3f} % ({wer_b['n_above_50_pct_wer']}) | |"
        )
    sim_a, sim_b = read(arm_a, cell, "similarity_results.json"), read(arm_b, cell, "similarity_results.json")
    if sim_a and sim_b:
        print(f"| {cell} | SIM | {sim_a['speaker_similarity_mean']:.2f} | {sim_b['speaker_similarity_mean']:.2f} | |")
