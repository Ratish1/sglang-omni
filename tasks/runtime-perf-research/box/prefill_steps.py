"""The talker's prefill steps in an events cell, from its scheduler_prefill_start and scheduler_prefill_end events
(requests of one step start within 1 ms of each other): the step's request count against its duration (first start to
last end), a least-squares line of duration on requests, and the steps that admit the pass's opening burst (the first
second of the cell) apart from the rest.

usage: python3 prefill_steps.py <cell> <boot> [<boot> ...]
"""

import glob
import json
import statistics
import sys
from collections import defaultdict

STEP_SPLIT_NS = 1_000_000
OPENING_S = 1.0
cell = sys.argv[1]
for boot in sys.argv[2:]:
    starts, ends = defaultdict(list), defaultdict(list)
    for path in glob.glob(f"{boot}/{cell}/events/*.jsonl"):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if '"talker_ar"' not in line:
                    continue
                record = json.loads(line)
                if record["event_name"] == "scheduler_prefill_start":
                    starts[record["request_id"]].append(record["timestamp_ns"])
                elif record["event_name"] == "scheduler_prefill_end":
                    ends[record["request_id"]].append(record["timestamp_ns"])
    admitted = sorted((min(starts[rid]), max(ends[rid])) for rid in starts if rid in ends)
    steps = []
    for start, end in admitted:
        if steps and start - steps[-1][0] <= STEP_SPLIT_NS:
            steps[-1][1] = max(steps[-1][1], end)
            steps[-1][2] += 1
        else:
            steps.append([start, end, 1])
    origin = steps[0][0]
    rows = [((end - start) / 1e6, count, (start - origin) / 1e9) for start, end, count in steps]
    sizes = [count for _, count, _ in rows]
    durations = [ms for ms, _, _ in rows]
    mean_size, mean_ms = statistics.mean(sizes), statistics.mean(durations)
    slope = sum((s - mean_size) * (d - mean_ms) for d, s, _ in rows) / max(sum((s - mean_size) ** 2 for s in sizes), 1e-9)
    print(f"{boot}: {len(rows)} prefill steps, {sum(sizes)} requests, requests a step mean {mean_size:.2f} max "
          f"{max(sizes)}, duration ms = {mean_ms - slope * mean_size:.1f} + {slope:.2f} x requests")
    by_size = defaultdict(list)
    for ms, count, _ in rows:
        by_size[min(count, 12)].append(ms)
    print("  by requests (12 = 12 or more): " + ", ".join(
        f"{size}: {len(v)} x {statistics.median(v):.1f}" for size, v in sorted(by_size.items())))
    opening = [(ms, count) for ms, count, t in rows if t < OPENING_S]
    print(f"  opening {OPENING_S:.0f} s: {len(opening)} steps, requests {[c for _, c in opening]}, "
          f"ms {[round(ms) for ms, _ in opening]}")
