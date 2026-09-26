"""Per-step device time of one torch profiler trace: the sum of kernel spans, their union, and
per kernel name the part of its span that starts under the previous kernel on the same stream
(programmatic dependent launch starts a dependent kernel inside its producer's span, so a sum of
spans overcounts it). usage: trace_union.py <trace.json.gz> [steps]"""

import collections
import gzip
import json
import sys

path = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
with gzip.open(path, "rt") as handle:
    events = json.load(handle)["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel"]
kernels.sort(key=lambda e: e["ts"])
union = 0.0
cur_start = cur_end = None
for e in kernels:
    start, end = e["ts"], e["ts"] + e["dur"]
    if cur_end is None or start > cur_end:
        if cur_end is not None:
            union += cur_end - cur_start
        cur_start, cur_end = start, end
    else:
        cur_end = max(cur_end, end)
if cur_end is not None:
    union += cur_end - cur_start
total = sum(e["dur"] for e in kernels)
window = kernels[-1]["ts"] + kernels[-1]["dur"] - kernels[0]["ts"]
rows = collections.defaultdict(lambda: [0.0, 0, 0.0])
prev_end = {}
for e in kernels:
    stream = e.get("args", {}).get("stream", 0)
    start, dur = e["ts"], e["dur"]
    hidden = max(0.0, min(prev_end.get(stream, start), start + dur) - start)
    row = rows[e["name"][:70]]
    row[0] += dur
    row[1] += 1
    row[2] += hidden
    prev_end[stream] = max(prev_end.get(stream, 0), start + dur)
print(
    f"kernels {len(kernels) / steps:.0f}/step sum {total / 1e3 / steps:.2f} ms/step "
    f"union {union / 1e3 / steps:.2f} ms/step window {window / 1e3 / steps:.2f} ms/step "
    f"streams {sorted(prev_end)}"
)
for name, (dur, count, hidden) in sorted(rows.items(), key=lambda r: -r[1][0])[:22]:
    print(
        f"{dur / 1e3 / steps:7.3f} ms/step {count / steps:5.0f}/step hidden {hidden / 1e3 / steps:6.3f}  {name}"
    )
