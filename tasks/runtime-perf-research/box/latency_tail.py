"""Whether a pass's latency tail is its replies' length or its speed: per boot and load, over the clean passes, latency
p99 and the mean audio length and latency per audio second of the slowest 1 % of requests, then the same prompts'
values in the other arm.

usage: python3 latency_tail.py <root> <load> <arm a boots a1,a2> <arm b boots b1,b2>
"""

import json
import statistics
import sys
from pathlib import Path

root, load = Path(sys.argv[1]), sys.argv[2]
arms = [sys.argv[3].split(","), sys.argv[4].split(",")]


def entries(boot):
    rows = []
    for rep in ("rep0", "rep1"):
        for entry in json.loads((root / boot / f"{load}_clean/bench/{rep}/generated.json").read_text()):
            if entry["is_success"] and entry.get("audio_duration_s"):
                rows.append(entry)
    return rows


by_arm = []
for arm in arms:
    rows = [entry for boot in arm for entry in entries(boot)]
    by_arm.append(rows)
    latencies = sorted(entry["latency_s"] for entry in rows)
    cut = latencies[int(0.99 * (len(latencies) - 1))]
    tail = [entry for entry in rows if entry["latency_s"] >= cut]
    print(f"{'+'.join(arm)}: {len(rows)} requests, latency p99 {cut * 1e3:.0f} ms, "
          f"latency per audio s (all) {statistics.mean(e['latency_s'] / e['audio_duration_s'] for e in rows):.4f}; "
          f"slowest 1 % ({len(tail)}): audio {statistics.mean(e['audio_duration_s'] for e in tail):.2f} s, "
          f"latency {statistics.mean(e['latency_s'] for e in tail) * 1e3:.0f} ms, "
          f"per audio s {statistics.mean(e['latency_s'] / e['audio_duration_s'] for e in tail):.4f}")
    for entry in sorted(tail, key=lambda e: -e["latency_s"])[:6]:
        print(f"    {entry['sample_id'][:48]} latency {entry['latency_s'] * 1e3:.0f} audio {entry['audio_duration_s']:.2f}")
duration_bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 1e9)]
print("latency mean (ms) by audio length, arm a / arm b:")
for low, high in duration_bins:
    cells = []
    for rows in by_arm:
        sel = [e["latency_s"] for e in rows if low <= e["audio_duration_s"] < high]
        cells.append(f"{statistics.mean(sel) * 1e3:.0f} (n {len(sel)})" if sel else "-")
    print(f"  {low}-{high if high < 1e9 else 'inf'} s: {cells[0]} / {cells[1]}")
