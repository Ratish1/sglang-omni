"""What the AR thread does inside one probe range (default mr.collect), from nsys reports taken with the probe: per
range, its wall time, the CUDA runtime calls made inside it by name (count and time), GIL waits, and the rest (Python).
Ranges nested inside it (mr.host_tokens, mr.emit) are left out of every column, as the ledgers' innermost attribution.

usage: python3 ar_collect_calls_c20261018.py NAME=REPORT.sqlite [NAME=REPORT.sqlite ...] [--range mr.collect]"""

import argparse
import bisect
import collections
import sqlite3

GIL_WAITS = ("pthread_cond_timedwait", "pthread_cond_clockwait")


def covered(intervals, a, b):
    return sum(max(0, min(end, b) - max(start, a)) for start, end in intervals)


def analyse(name, path, wanted):
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    ranges = collections.defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
            "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType in (59, 60) and end is not null"):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith(("loop.", "mr.")):
            ranges[tid].append((start, end, label))
    engine = next(tid for tid, rows in ranges.items() if any(label.startswith("loop.step") for _, _, label in rows))
    outer = sorted((s, e) for s, e, label in ranges[engine] if label == wanted)
    # note: ranges on one thread nest, so another mr.* range starting inside one of ours is nested in it.
    inner = sorted((s, e) for s, e, label in ranges[engine] if label != wanted and label.startswith("mr."))
    inner_starts = [s for s, _ in inner]
    calls = sorted((s, e, strings.get(n, "").split("_v")[0]) for s, e, n in db.execute(
        "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ?", (engine,)))
    call_starts = [c[0] for c in calls]
    gil = sorted((s, e) for s, e, n in db.execute("select start, end, nameId from OSRT_API where globalTid = ?",
                                                  (engine,)) if strings.get(n, "").startswith(GIL_WAITS))
    gil_starts = [s for s, _ in gil]
    by_call, wall_total, gil_total = collections.Counter(), 0, 0
    counts = collections.Counter()
    for a, b in outer:
        lo, hi = bisect.bisect_left(inner_starts, a), bisect.bisect_left(inner_starts, b)
        nested = inner[lo:hi]
        own = (b - a) - sum(e - s for s, e in nested)
        wall_total += own
        for s, e, call in calls[bisect.bisect_left(call_starts, a):bisect.bisect_left(call_starts, b)]:
            if any(ns <= s < ne for ns, ne in nested):
                continue
            by_call[call] += e - s
            counts[call] += 1
        waits = gil[max(0, bisect.bisect_left(gil_starts, a) - 1):bisect.bisect_left(gil_starts, b)]
        gil_total += covered(waits, a, b) - sum(covered(waits, s, e) for s, e in nested)
    n = len(outer)
    api = sum(by_call.values())
    print(f"\n# {name}: {wanted} x {n}, own time per range {wall_total / n / 1e3:.1f} us: CUDA calls "
          f"{api / n / 1e3:.1f} us ({sum(counts.values()) / n:.2f} calls), GIL waits {gil_total / n / 1e3:.1f} us, "
          f"the rest {(wall_total - api - gil_total) / n / 1e3:.1f} us")
    for call, t in by_call.most_common(10):
        print(f"  {call:32} {counts[call] / n:6.2f} per range, {t / counts[call] / 1e3:7.1f} us each, "
              f"{t / n / 1e3:7.1f} us per range")


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+")
parser.add_argument("--range", default="mr.collect")
arguments = parser.parse_args()
for report in arguments.reports:
    label, _, path = report.partition("=")
    analyse(label, path, arguments.range)
