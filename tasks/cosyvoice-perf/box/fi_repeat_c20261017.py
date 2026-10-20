"""Is #2897's unseeded tail rise real? Reads the three main against F pairs (p_fi, p_fi2, p_fi3) and the two main
against main pairs (aa_u2, aa_u3) of 10-17, unseeded c16 and c64, both orders each:

1. Per order sample, the change of every p99 row and of the means (F against main; second server against first for
   main main), so F's spread can be read against main main's on the same cards and day.
2. Pooled over every request of an arm (3 pairs x 2 orders x 1,088 for F and its main, 2 x 2 x 1,088 per main main
   arm): first audio, latency and inter chunk at the median, p95, p99 and p99.5, F's change against main and main
   main's own.

usage: python3 fi_repeat_c20261017.py <copy of /data/c20261017> (artifacts/cosyvoice-c20261017)"""

import json
import statistics
import sys
from pathlib import Path

ORDERS = (("a", ("gpu0", "gpu1")), ("b", ("gpu1", "gpu0")))
ROWS = (("throughput_qps", "req/s"), ("audio_ttfp_mean_s", "first audio mean"), ("audio_ttfp_p99_s", "first audio p99"),
        ("inter_chunk_mean_s", "inter chunk mean"), ("inter_chunk_p99_s", "inter chunk p99"),
        ("latency_mean_s", "latency mean"), ("latency_p99_s", "latency p99"))


def cell_dir(root, pair, order, gpu, cell):
    return root / pair / order / gpu / cell / "speed_results.json"


def per_order(root, pairs, cell):
    changes = {label: [] for _, label in ROWS}
    for pair in pairs:
        for order, (base, arm) in ORDERS:
            first = json.loads(cell_dir(root, pair, order, base, cell).read_text())["summary"]
            second = json.loads(cell_dir(root, pair, order, arm, cell).read_text())["summary"]
            for key, label in ROWS:
                changes[label].append(100 * (second[key] - first[key]) / first[key])
    return changes


def pooled(root, pairs, cell):
    arms = ([], [])
    for pair in pairs:
        for order, gpus in ORDERS:
            for side, gpu in enumerate(gpus):
                rows = json.loads(cell_dir(root, pair, order, gpu, cell).read_text())["per_request"]
                arms[side].extend(row for row in rows if row["is_success"])
    return arms


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def metrics(rows):
    first_audio = [row["audio_ttfp_s"] for row in rows]
    latency = [row["latency_s"] for row in rows]
    gaps = [gap for row in rows for gap in (row["inter_chunk_s"] or [])]
    out = {}
    for name, values in (("first audio", first_audio), ("latency", latency), ("inter chunk", gaps)):
        for label, fraction in (("median", 0.5), ("p95", 0.95), ("p99", 0.99), ("p99.5", 0.995)):
            out[f"{name} {label}"] = quantile(values, fraction)
        out[f"{name} mean"] = statistics.mean(values)
    return out


root = Path(sys.argv[1])
groups = (("F against main", ("p_fi", "p_fi2", "p_fi3")), ("main against main", ("aa_u2", "aa_u3")))
for cell in ("c16", "c64"):
    print(f"\n## {cell}: change per order sample (%)")
    for name, pairs in groups:
        for label, values in per_order(root, pairs, cell).items():
            print(f"{name:18} {label:17} " + " ".join(f"{v:+6.1f}" for v in values)
                  + f"   mean {statistics.mean(values):+5.1f}")
    print(f"\n## {cell}: pooled over every request of an arm")
    pooled_metrics = {}
    for name, pairs in groups:
        first, second = pooled(root, pairs, cell)
        pooled_metrics[name] = (metrics(first), metrics(second), len(first), len(second))
    print("| metric | main (F pairs) | F | change | main main: first | second | change |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for key in pooled_metrics["F against main"][0]:
        base, arm, _, _ = pooled_metrics["F against main"]
        aa_first, aa_second, _, _ = pooled_metrics["main against main"]
        print(f"| {key} | {base[key]:.3f} | {arm[key]:.3f} | {100 * (arm[key] - base[key]) / base[key]:+.1f} % "
              f"| {aa_first[key]:.3f} | {aa_second[key]:.3f} | {100 * (aa_second[key] - aa_first[key]) / aa_first[key]:+.1f} % |")
    print("requests: " + ", ".join(f"{name} {n1} / {n2}" for name, (_, _, n1, n2) in pooled_metrics.items()))
