import glob
import json
import statistics
import sys


def requests(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        for request in json.load(open(path))["per_request"]:
            if request["is_success"]:
                rows.append(request)
    return rows


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


for label, pattern in (("main", sys.argv[1]), ("PR", sys.argv[2])):
    rows = requests(pattern)
    latency = [r["latency_s"] for r in rows]
    after_first = [r["latency_s"] - r["audio_ttfp_s"] for r in rows]
    duration = [r["audio_duration_s"] for r in rows]
    rtf = [r["rtf"] for r in rows]
    p99 = quantile(latency, 0.99)
    tail = [r for r in rows if r["latency_s"] >= p99]
    print(label, "n", len(rows))
    print("  latency p50/p95/p99", [round(quantile(latency, q), 3) for q in (0.5, 0.95, 0.99)])
    print("  after first audio p50/p95/p99", [round(quantile(after_first, q), 3) for q in (0.5, 0.95, 0.99)])
    print("  RTF p50/p95/p99", [round(quantile(rtf, q), 3) for q in (0.5, 0.95, 0.99)])
    print("  tail >= p99: n", len(tail), "audio s mean", round(statistics.mean(r["audio_duration_s"] for r in tail), 2),
          "first audio mean", round(statistics.mean(r["audio_ttfp_s"] for r in tail), 3),
          "after first mean", round(statistics.mean(r["latency_s"] - r["audio_ttfp_s"] for r in tail), 3),
          "rtf mean", round(statistics.mean(r["rtf"] for r in tail), 3))
    chunk_gaps = [gap for r in rows for gap in (r["inter_chunk_s"] or [])[1:]]
    print("  inter-chunk gap after the first p50/p99", [round(quantile(chunk_gaps, q), 3) for q in (0.5, 0.99)])
    long_rows = [r for r in rows if r["audio_duration_s"] >= quantile(duration, 0.9)]
    print("  longest 10 % clips: latency mean", round(statistics.mean(r["latency_s"] for r in long_rows), 3),
          "first audio mean", round(statistics.mean(r["audio_ttfp_s"] for r in long_rows), 3),
          "rtf mean", round(statistics.mean(r["rtf"] for r in long_rows), 3), "audio s", round(statistics.mean(r["audio_duration_s"] for r in long_rows), 2))
