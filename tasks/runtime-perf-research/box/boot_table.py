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
    ("first audio mean (ms)", "audio_ttfp_mean_s", 1000, "{:.0f}"),
    ("first audio p99 (ms)", "audio_ttfp_p99_s", 1000, "{:.0f}"),
    ("inter chunk mean (ms)", "inter_chunk_mean_s", 1000, "{:.0f}"),
    ("inter chunk p99 (ms)", "inter_chunk_p99_s", 1000, "{:.0f}"),
]


def boot_mean(pattern, key):
    values = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as source:
            value = json.load(source)["summary"].get(key)
        if value is not None:
            values.append(value)
        else:
            pass
    return statistics.mean(values) if values else None


run, cell, label, main_boots, branch_boots = sys.argv[1:6]
reps = sys.argv[6] if len(sys.argv) > 6 else "rep*"
print(f"| {label} | main | this PR | change |")
print("|---|---|---|---|")
for name, key, scale, fmt in ROWS:
    main = [
        boot_mean(f"{run}/{boot}/{cell}/bench/{reps}/speed_results.json", key)
        for boot in main_boots.split(",")
    ]
    branch = [
        boot_mean(f"{run}/{boot}/{cell}/bench/{reps}/speed_results.json", key)
        for boot in branch_boots.split(",")
    ]
    if None in main or None in branch:
        continue
    change = (
        (statistics.mean(branch) - statistics.mean(main)) / statistics.mean(main) * 100
    )
    print(
        f"| {name} | {' / '.join(fmt.format(v * scale) for v in main)} | {' / '.join(fmt.format(v * scale) for v in branch)} | {change:+.1f} % |"
    )
