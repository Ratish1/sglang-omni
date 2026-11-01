"""Latency and first audio paired by prompt between two arms at one load, over every clean pass: each prompt's mean in
each arm, the per-prompt difference (mean with a 95 % interval), the same for latency per second of audio (the
reply-length-free form), and each arm's pooled p99. Prompts are matched by sample id.

usage: python3 latency_paired.py <root> <load> <arm a boots a1,a2> <arm b boots b1,b2>
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root, load = Path(sys.argv[1]), sys.argv[2]
arms = [sys.argv[3].split(","), sys.argv[4].split(",")]


def per_prompt(boots):
    rows = defaultdict(list)
    for boot in boots:
        for rep in ("rep0", "rep1"):
            requests = json.loads((root / boot / f"{load}_clean/bench/{rep}/speed_results.json").read_text())["per_request"]
            for r in requests:
                if r["is_success"] and r["audio_duration_s"] and r["audio_ttfp_s"] is not None:
                    rows[r["id"]].append((r["latency_s"] * 1e3, r["audio_ttfp_s"] * 1e3, r["audio_duration_s"]))
    return rows


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


a, b = per_prompt(arms[0]), per_prompt(arms[1])
shared = sorted(set(a) & set(b))
for label, pick in (("latency (ms)", lambda v: v[0]), ("first audio (ms)", lambda v: v[1]),
                    ("latency per audio second (ms/s)", lambda v: v[0] / v[2])):
    diffs = [statistics.mean(pick(v) for v in b[p]) - statistics.mean(pick(v) for v in a[p]) for p in shared]
    mean = statistics.mean(diffs)
    half = 1.96 * statistics.stdev(diffs) / len(diffs) ** 0.5
    base = statistics.mean(statistics.mean(pick(v) for v in a[p]) for p in shared)
    pooled = [quantile([pick(v) for p in shared for v in arm[p]], 0.99) for arm in (a, b)]
    print(f"{label}: {len(shared)} prompts, arm a mean {base:.1f}, paired difference {mean:+.2f} "
          f"({mean / base * 100:+.2f} %, 95 % {mean - half:+.2f} to {mean + half:+.2f}); pooled p99 "
          f"{pooled[0]:.1f} against {pooled[1]:.1f}")
