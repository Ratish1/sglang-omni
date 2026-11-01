"""First audio of each c64 clean pass split into the opening burst (the first 64 requests, dispatched together at the
pass start in sample order) and the rest, per arm over its boots.

usage: python3 burst_split.py <root> <arm>=<boot>,<boot> [<arm>=<boot>,<boot> ...]
"""

import json
import statistics
import sys
from pathlib import Path

BURST = 64
root = Path(sys.argv[1])


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


for spec in sys.argv[2:]:
    arm, boots = spec.split("=")
    head, rest = [], []
    for boot in boots.split(","):
        for rep in ("rep0", "rep1"):
            requests = json.loads((root / boot / f"c64_clean/bench/{rep}/speed_results.json").read_text())["per_request"]
            values = [r["audio_ttfp_s"] * 1e3 for r in requests if r["is_success"] and r["audio_ttfp_s"] is not None]
            head += values[:BURST]
            rest += values[BURST:]
    print(f"{arm}: first {BURST} mean {statistics.mean(head):.0f} p95 {quantile(head, 0.95):.0f} | rest mean "
          f"{statistics.mean(rest):.0f} p95 {quantile(rest, 0.95):.0f} p99 {quantile(rest, 0.99):.0f} | all p95 "
          f"{quantile(head + rest, 0.95):.0f} p99 {quantile(head + rest, 0.99):.0f}")
