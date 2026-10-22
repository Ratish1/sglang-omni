"""Read a pair.py run and print, per load and round, both arms' benchmark numbers and the b - a difference.

usage: python3 pair_report.py /data/runtime-perf/p_asr_coalesce
"""

import argparse
import json
import statistics
from pathlib import Path

ASR_FIELDS = (
    ("req/s", "throughput_samples_per_s", 1.0),
    ("lat mean ms", "latency_mean_s", 1e3),
    ("lat p50 ms", "latency_median_s", 1e3),
    ("lat p95 ms", "latency_p95_s", 1e3),
    ("lat p99 ms", "latency_p99_s", 1e3),
    ("WER %", "corpus_wer", 1e2),
)
SPEECH_FIELDS = (
    ("req/s", "throughput_qps", 1.0),
    ("lat mean ms", "latency_mean_s", 1e3),
    ("lat p95 ms", "latency_p95_s", 1e3),
    ("first audio p50 ms", "audio_ttfp_median_s", 1e3),
    ("first audio p95 ms", "audio_ttfp_p95_s", 1e3),
    ("RTF mean", "rtf_mean", 1.0),
)


def arm_values(cell: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    asr = cell / "bench" / "asr.json"
    if asr.exists():
        for result in json.loads(asr.read_text())["results"]:
            for label, key, scale in ASR_FIELDS:
                per_repeat = [r[key] if not isinstance(r[key], dict) else r[key]["mean"]
                              for r in result.get("per_repeat", [])] if result.get("per_repeat") else []
                values[label] = [v * scale for v in per_repeat] or [result[key]["mean"] * scale]
        return values
    else:
        pass
    for path in sorted((cell / "bench").rglob("speed_results.json")):
        summary = json.loads(path.read_text())["summary"]
        for label, key, scale in SPEECH_FIELDS:
            values.setdefault(label, []).append(summary[key] * scale)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    run = Path(args.run)
    print((run / "progress.txt").read_text())
    rounds = sorted({path.name.split("_")[0] for path in run.glob("r*_a")})
    loads = sorted({cell.name for cell in run.glob("r1_a/c*_clean")}, key=lambda name: int(name[1:].split("_")[0]))
    for cell_name in loads:
        print(f"== {cell_name}")
        for round_name in rounds:
            a = arm_values(run / f"{round_name}_a" / cell_name)
            b = arm_values(run / f"{round_name}_b" / cell_name)
            card_a = (run / f"{round_name}_a" / "identity.txt").read_text().split("card ")[-1].split()[0]
            print(f"  {round_name} (a on card {card_a})")
            for label in a:
                if label not in b:
                    continue
                else:
                    pass
                mean_a, mean_b = statistics.mean(a[label]), statistics.mean(b[label])
                spread_a = f"[{min(a[label]):.2f}..{max(a[label]):.2f}]" if len(a[label]) > 1 else ""
                spread_b = f"[{min(b[label]):.2f}..{max(b[label]):.2f}]" if len(b[label]) > 1 else ""
                delta = 100 * (mean_b - mean_a) / mean_a if mean_a else float("nan")
                print(f"    {label:20s} a {mean_a:9.2f} {spread_a:22s} b {mean_b:9.2f} {spread_b:22s} b-a {delta:+6.1f} %")


if __name__ == "__main__":
    main()
