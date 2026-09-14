"""Compare one streaming c16 speed_results.json with the fixed A baseline.

Usage: python compare_to_a.py --model moss_tts_local --speed-results <run>/speed_results.json
"""

import argparse
import json
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent / "baselines" / "stream_en_c16.json"

METRICS = (
    ("completed", "completed_requests"),
    ("failed", "failed_requests"),
    ("req/s", "throughput_qps"),
    ("audio s/s", "audio_throughput_s_per_s"),
    ("RTF mean", "rtf_mean"),
    ("RTF p95", "rtf_p95"),
    ("RTF p99", "rtf_p99"),
    ("first audio mean s", "audio_ttfp_mean_s"),
    ("first audio p50 s", "audio_ttfp_median_s"),
    ("first audio p95 s", "audio_ttfp_p95_s"),
    ("first audio p99 s", "audio_ttfp_p99_s"),
    ("inter chunk mean s", "inter_chunk_mean_s"),
    ("C50", "c50"),
    ("C100", "c100"),
    ("latency mean s", "latency_mean_s"),
)


def _cell(value):
    return "-" if value is None else f"{value:g}"


def _delta(a, b):
    if a is None or b is None:
        return "-", "-"
    diff = b - a
    percent = f"{100.0 * diff / a:+.1f}%" if a else "-"
    return f"{diff:+.4g}", percent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("moss_tts_local", "qwen3_tts")
    )
    parser.add_argument("--speed-results", required=True)
    args = parser.parse_args()

    baseline = json.loads(BASELINES.read_text())[args.model]
    run = json.loads(Path(args.speed_results).read_text())["summary"]
    a = baseline["a"]["summary"]
    pr1 = baseline["pr1_head"]["summary"]

    print(f"{args.model}")
    print(f"A: {baseline['a']['source']}")
    print(f"B: {args.speed_results}")
    print()
    print("| metric | A | PR 1 head | B | B - A | B vs A |")
    print("|---|---|---|---|---|---|")
    for name, key in METRICS:
        diff, percent = _delta(a.get(key), run.get(key))
        print(
            f"| {name} | {_cell(a.get(key))} | {_cell(pr1.get(key))} | "
            f"{_cell(run.get(key))} | {diff} | {percent} |"
        )


if __name__ == "__main__":
    main()
