import glob
import json
import statistics
import sys

KEYS = [
    ("QPS", "throughput_qps"),
    ("RTF mean", "rtf_mean"),
    ("latency median", "latency_median_s"),
    ("latency p95", "latency_p95_s"),
    ("latency p99", "latency_p99_s"),
    ("first audio median", "audio_ttfp_median_s"),
    ("first audio p95", "audio_ttfp_p95_s"),
    ("first audio p99", "audio_ttfp_p99_s"),
]


def passes(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        summary = json.load(open(path))["summary"]
        if summary.get("max_token_hits", 0) == 0:
            rows.append(summary)
    return rows


run, cell = sys.argv[1], sys.argv[2]
arms = {name: passes(f"{run}_{name}/{cell}/bench/rep[12]/speed_results.json") for name in ("r1_a", "r1_b", "r2_a", "r2_b")}
print(f"{run} {cell}: passes per boot", {k: len(v) for k, v in arms.items()})
print("| metric | A/A delta order 1 | A/A delta order 2 | two-order mean | sd of boot means (% of mean) |")
print("|---|---|---|---|---|")
for name, key in KEYS:
    boot_means = {k: statistics.mean(s[key] for s in v) for k, v in arms.items()}
    delta1 = (boot_means["r1_b"] - boot_means["r1_a"]) / boot_means["r1_a"] * 100
    delta2 = (boot_means["r2_b"] - boot_means["r2_a"]) / boot_means["r2_a"] * 100
    values = list(boot_means.values())
    sd = statistics.stdev(values) / statistics.mean(values) * 100
    print(f"| {name} | {delta1:+.1f} % | {delta2:+.1f} % | {(delta1 + delta2) / 2:+.1f} % | {sd:.1f} % |")
