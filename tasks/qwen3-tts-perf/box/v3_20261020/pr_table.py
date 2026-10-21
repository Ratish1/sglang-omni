"""One pair.py run as a PR body's table: arm a is main, arm b the PR.

usage: python3 pr_table.py LABEL   (under PAIR_ROOT, default /data/v3_20261020/pairs)

Rows: latency median, p95 and p99; first audio median, p95 and p99 (streaming only); QPS; RTF; WER over the samples
under 50 % and the count over it; speaker similarity; the arm's card memory after the benchmark.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/v3_20261020/pairs"))


def last_float(pattern: str, text: str) -> float | None:
    found = re.findall(pattern, text)
    return float(found[-1]) if found else None


def arm(label: str, name: str) -> dict:
    directory = ROOT / label / name
    card = int((directory / "tree.txt").read_text().splitlines()[0].split()[2])
    summary = json.load(open(directory / "bench" / "speed_results.json"))["summary"]
    values = {key: summary.get(key) for key in summary}
    transcribe = (directory / "score--transcribe-only.log").read_text(errors="ignore")
    values["wer_under_half"] = last_float(r"WER corpus \(excl >50%\):\s+([0-9.]+)", transcribe)
    wer_path = directory / "bench" / "wer_results.json"
    values["over_half"] = sum(
        sample["wer"] is not None and float(sample["wer"]) > 0.5
        for sample in json.load(open(wer_path))["per_sample"]
    ) if wer_path.exists() else None
    similarity = (directory / "score--similarity-only.log").read_text(errors="ignore")
    values["similarity"] = last_float(r"speaker similarity: ([0-9.]+)", similarity)
    memory = (directory / "gpu_after_bench.txt").read_text().splitlines()[card]
    values["memory_mib"] = int(memory.split(",")[1].split()[0])
    return values


def percent(main: float, pr: float) -> str:
    return f"{100 * (pr / main - 1):+.1f} %"


def main() -> None:
    label = sys.argv[1]
    a, b = arm(label, "a"), arm(label, "b")
    print("| | main | this PR | delta |")
    print("| --- | --- | --- | --- |")
    for key, name in (("latency_median_s", "latency median"), ("latency_p95_s", "latency p95"),
                      ("latency_p99_s", "latency p99")):
        print(f"| {name} | {a[key]:.3f} s | {b[key]:.3f} s | {percent(a[key], b[key])} |")
    if a.get("audio_ttfp_median_s") is not None:
        for key, name in (("audio_ttfp_median_s", "first audio median"), ("audio_ttfp_p95_s", "first audio p95"),
                          ("audio_ttfp_p99_s", "first audio p99")):
            print(f"| {name} | {1e3 * a[key]:.1f} ms | {1e3 * b[key]:.1f} ms | {percent(a[key], b[key])} |")
    else:
        pass
    print(f"| QPS | {a['throughput_qps']:.2f} | {b['throughput_qps']:.2f} | "
          f"{percent(a['throughput_qps'], b['throughput_qps'])} |")
    print(f"| RTF | {a['rtf_mean']:.4f} | {b['rtf_mean']:.4f} | {percent(a['rtf_mean'], b['rtf_mean'])} |")
    print(f"| WER below 50 % | {100 * a['wer_under_half']:.2f} % | {100 * b['wer_under_half']:.2f} % | "
          f"{100 * (b['wer_under_half'] - a['wer_under_half']):+.2f} pt |")
    print(f"| samples over 50 % WER | {a['over_half']} | {b['over_half']} | |")
    print(f"| speaker similarity | {a['similarity']:.2f} | {b['similarity']:.2f} | "
          f"{b['similarity'] - a['similarity']:+.2f} |")
    print(f"| card memory | {a['memory_mib']:,} MiB | {b['memory_mib']:,} MiB | |")


if __name__ == "__main__":
    main()
