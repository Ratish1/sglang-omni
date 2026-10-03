"""CUDA runtime calls inside one kind of NVTX range, by call name: count per range and time.

usage: python nsys_range_api_calls.py serve.sqlite "sched.launch decode" [more sqlite files]
"""

import sqlite3
import sys
from collections import defaultdict


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


kind = sys.argv[2]
for path in [sys.argv[1]] + sys.argv[3:]:
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    ranges = []
    for start, end, thread, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS where end is not null"
    ):
        label = text if text is not None else strings.get(text_id, "")
        if " ".join(label.split(" ")[:2]) == kind:
            ranges.append((start, end, thread))
        else:
            pass
    threads = {thread for _, _, thread in ranges}
    calls = defaultdict(list)
    for start, end, thread, name_id in db.execute(
        "select start, end, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        if thread in threads:
            calls[thread].append((start, end, strings.get(name_id, str(name_id))))
        else:
            pass
    for thread in calls:
        calls[thread].sort()
    totals = defaultdict(lambda: [0, 0.0, 0.0])
    wall = 0.0
    for start, end, thread in ranges:
        wall += (end - start) / 1e6
        for call_start, call_end, name in calls[thread]:
            if call_start >= start and call_end <= end:
                row = totals[name]
                row[0] += 1
                duration = (call_end - call_start) / 1e6
                row[1] += duration
                row[2] = max(row[2], duration)
            else:
                pass
    print(f"== {path}: {len(ranges)} x '{kind}', wall {wall:.0f} ms")
    print(
        f"  {'call':40s} {'per range':>9s} {'ms total':>9s} {'ms/range':>9s} {'max ms':>8s}"
    )
    for name, (count, total, longest) in sorted(
        totals.items(), key=lambda item: -item[1][1]
    )[:12]:
        print(
            f"  {name[:40]:40s} {count / max(len(ranges), 1):9.2f} {total:9.1f}"
            f" {total / max(len(ranges), 1):9.3f} {longest:8.2f}"
        )
