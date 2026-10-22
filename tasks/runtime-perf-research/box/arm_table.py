import glob
import json
import sys

SPEECH_KEYS = [
    ("QPS", "throughput_qps", 1, "{:.2f}"),
    ("RTF mean", "rtf_mean", 1, "{:.3f}"),
    ("latency median (ms)", "latency_median_s", 1000, "{:.0f}"),
    ("latency p95 (ms)", "latency_p95_s", 1000, "{:.0f}"),
    ("latency p99 (ms)", "latency_p99_s", 1000, "{:.0f}"),
    ("first audio median (ms)", "audio_ttfp_median_s", 1000, "{:.1f}"),
    ("first audio p95 (ms)", "audio_ttfp_p95_s", 1000, "{:.1f}"),
    ("first audio p99 (ms)", "audio_ttfp_p99_s", 1000, "{:.1f}"),
]
ASR_KEYS = [
    ("QPS", "throughput_samples_per_s", 1, "{:.1f}"),
    ("RTF mean", "rtf_mean", 1, "{:.4f}"),
    ("latency median (ms)", "latency_median_s", 1000, "{:.1f}"),
    ("latency p95 (ms)", "latency_p95_s", 1000, "{:.1f}"),
    ("latency p99 (ms)", "latency_p99_s", 1000, "{:.1f}"),
    ("corpus WER (%)", "corpus_wer", 100, "{:.2f}"),
]


def load_speech(pattern, drop_max_token):
    summaries = []
    for path in sorted(glob.glob(pattern)):
        summary = json.load(open(path))["summary"]
        if drop_max_token and summary.get("max_token_hits", 0) > 0:
            print(f"  dropped {path}: {summary['max_token_hits']} max-token replies")
            continue
        summaries.append(summary)
    return summaries


def load_asr(pattern):
    summaries = []
    for path in sorted(glob.glob(pattern)):
        for result in json.load(open(path))["results"]:
            summaries.append({key: value["mean"] for key, value in result.items() if isinstance(value, dict) and "mean" in value})
    return summaries


def mean(values):
    return sum(values) / len(values)


kind, label_a, pattern_a, label_b, pattern_b = sys.argv[1:6]
if kind == "asr":
    arm_a, arm_b, keys = load_asr(pattern_a), load_asr(pattern_b), ASR_KEYS
else:
    drop = kind == "tts"
    arm_a, arm_b, keys = load_speech(pattern_a, drop), load_speech(pattern_b, drop), SPEECH_KEYS
print(f"passes: a {len(arm_a)}, b {len(arm_b)}")
print(f"| metric | {label_a} | {label_b} | change |")
print("|---|---|---|---|")
for name, key, scale, fmt in keys:
    value_a = mean([s[key] for s in arm_a]) * scale
    value_b = mean([s[key] for s in arm_b]) * scale
    change = (value_b - value_a) / value_a * 100
    print(f"| {name} | {fmt.format(value_a)} | {fmt.format(value_b)} | {change:+.1f} % |")
