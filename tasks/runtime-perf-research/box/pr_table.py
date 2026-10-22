import glob
import json
import statistics
import sys

ROWS = [
    ("QPS", "throughput_qps", 1, "{:.2f}"),
    ("audio s/s", "audio_throughput_s_per_s", 1, "{:.1f}"),
    ("RTF mean", "rtf_mean", 1, "{:.3f}"),
    ("latency median (ms)", "latency_median_s", 1000, "{:.0f}"),
    ("latency p95 (ms)", "latency_p95_s", 1000, "{:.0f}"),
    ("latency p99 (ms)", "latency_p99_s", 1000, "{:.0f}"),
    ("first audio mean (ms)", "audio_ttfp_mean_s", 1000, "{:.1f}"),
    ("first audio p99 (ms)", "audio_ttfp_p99_s", 1000, "{:.1f}"),
    ("inter chunk mean (ms)", "inter_chunk_mean_s", 1000, "{:.1f}"),
    ("inter chunk p99 (ms)", "inter_chunk_p99_s", 1000, "{:.1f}"),
]


def boot_mean(pattern, key):
    values = []
    for path in sorted(glob.glob(pattern)):
        summary = json.load(open(path))["summary"]
        if summary.get("max_token_hits", 0) == 0 and summary.get(key) is not None:
            values.append(summary[key])
    return statistics.mean(values) if values else None


prefixes, cell, label = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
reps = sys.argv[4] if len(sys.argv) > 4 else "rep[12]"
print(f"| {label} | main | this PR | change |")
print("|---|---|---|---|")
for name, key, scale, fmt in ROWS:
    main = [boot_mean(f"{p}_r{r}_a/{cell}/bench/{reps}/speed_results.json", key) for p in prefixes for r in (1, 2)]
    branch = [boot_mean(f"{p}_r{r}_b/{cell}/bench/{reps}/speed_results.json", key) for p in prefixes for r in (1, 2)]
    if None in main or None in branch:
        continue
    main_mean = statistics.mean(main) * scale
    branch_mean = statistics.mean(branch) * scale
    change = (branch_mean - main_mean) / main_mean * 100
    main_text = " / ".join(fmt.format(v * scale) for v in main)
    branch_text = " / ".join(fmt.format(v * scale) for v in branch)
    print(f"| {name} | {main_text} | {branch_text} | {change:+.1f} % |")
