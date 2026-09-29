"""The A/B table of a run_ab_pairs.sh directory: both arms' values, then the delta.

Reads bench/speed_results.json, and where scored bench/wer_results.json and
bench/similarity_results.json, of every <round>_a and <round>_b directory.

usage: python ab_table.py OUT_DIR
"""

from __future__ import annotations

import json
import os
import sys

SPEED = (
    ("req/s", "throughput_qps"),
    ("audio s/s", "audio_throughput_s_per_s"),
    ("TTFC mean ms", "audio_ttfp_mean_s"),
    ("TTFC p99 ms", "audio_ttfp_p99_s"),
    ("inter chunk mean ms", "inter_chunk_mean_s"),
    ("inter chunk p99 ms", "inter_chunk_p99_s"),
    ("latency mean s", "latency_mean_s"),
    ("latency p99 s", "latency_p99_s"),
    ("RTF mean", "rtf_mean"),
    ("RTF p99", "rtf_p99"),
    ("audio mean s", "audio_duration_mean_s"),
)
MILLISECONDS = {
    "audio_ttfp_mean_s",
    "audio_ttfp_p99_s",
    "inter_chunk_mean_s",
    "inter_chunk_p99_s",
}


def load(path: str) -> dict[str, float]:
    if os.path.isfile(path):
        with open(path) as handle:
            return json.load(handle)["summary"]
    else:
        return {}


def main() -> None:
    out = sys.argv[1]
    rounds = sorted(
        {name[:-2] for name in os.listdir(out) if name[-2:] in ("_a", "_b")}
    )
    for name in rounds:
        arms = []
        for arm in ("a", "b"):
            bench = os.path.join(out, f"{name}_{arm}", "bench")
            values = dict(load(os.path.join(bench, "speed_results.json")))
            wer = load(os.path.join(bench, "wer_results.json"))
            sim = load(os.path.join(bench, "similarity_results.json"))
            if wer:
                values["WER %"] = 100 * wer["wer_corpus"]
                values["WER above 50%"] = wer["n_above_50_pct_wer"]
            else:
                pass
            if sim:
                values["SIM"] = sim["speaker_similarity_mean"]
            else:
                pass
            arms.append(values)
        print(f"\n{name}")
        print(f"  {'metric':<22}{'A':>12}{'B':>12}{'delta':>10}")
        rows = [(label, key) for label, key in SPEED] + [
            ("WER %", "WER %"),
            ("WER above 50%", "WER above 50%"),
            ("SIM", "SIM"),
        ]
        for label, key in rows:
            if key not in arms[0] or key not in arms[1]:
                continue
            else:
                pass
            scale = 1000.0 if key in MILLISECONDS else 1.0
            a, b = arms[0][key] * scale, arms[1][key] * scale
            delta = f"{100 * (b - a) / a:+.1f}%" if a else "-"
            print(f"  {label:<22}{a:>12.4g}{b:>12.4g}{delta:>10}")


if __name__ == "__main__":
    main()
