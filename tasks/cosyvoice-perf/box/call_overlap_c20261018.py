"""What other threads were inside the CUDA runtime while the AR thread made one kind of call (default
cudaEventCreateWithFlags), from nsys reports: per such call, the share of its duration covered by each other thread's
runtime calls, by call name, and how the call's duration splits by whether another thread's call overlapped it. A
runtime call that waits on the context lock shows another thread's call over most of its duration.

usage: python3 call_overlap_c20261018.py NAME=REPORT.sqlite [NAME=REPORT.sqlite ...] [--call cudaEventCreateWithFlags]"""

import argparse
import bisect
import collections
import sqlite3


def analyse(name, path, wanted):
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    engine = None
    for tid, text, text_id in db.execute(
            "select globalTid, text, textId from NVTX_EVENTS where eventType in (59, 60) and end is not null"):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith("loop.step"):
            engine = tid
            break
    # note: the probe marks each Python thread once, "thread name=N native=T", on that thread.
    thread_names = {}
    for tid, text, text_id in db.execute("select globalTid, text, textId from NVTX_EVENTS where eventType = 34"):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith("thread name="):
            thread_names[tid] = label.split(" ")[1].removeprefix("name=")
    for (tid,) in db.execute(
            "select distinct e.globalTid from NVTX_EVENTS e left join StringIds s on e.textId = s.id "
            "where coalesce(e.text, s.value) like 'voc.step %'"):
        thread_names[tid] = "vocoder"
    calls = sorted((s, e, tid, f"{thread_names.get(tid, tid)}: {strings.get(n, '').split('_v')[0]}")
                   for s, e, tid, n in db.execute("select start, end, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"))
    mine = [c for c in calls if c[2] == engine and c[3].endswith(f": {wanted}")]
    others = [c for c in calls if c[2] != engine]
    other_starts = [c[0] for c in others]
    longest = max((e - s for s, e, _, _ in others), default=0)
    covered_by = collections.Counter()
    with_overlap = without_overlap = 0
    durations_with, durations_without = [], []
    for s, e, _, _ in mine:
        lo = bisect.bisect_left(other_starts, s - longest)
        hi = bisect.bisect_left(other_starts, e)
        hit = False
        for os_, oe, _, oname in others[lo:hi]:
            overlap = min(e, oe) - max(s, os_)
            if overlap > 0:
                covered_by[oname] += overlap
                hit = True
        (durations_with if hit else durations_without).append(e - s)
    total = sum(e - s for s, e, _, _ in mine)
    print(f"\n# {name}: {wanted} on the AR thread x {len(mine)}, {total / max(len(mine), 1) / 1e3:.1f} us each")
    for durations, title in ((durations_with, "another thread's call overlaps"), (durations_without, "none overlaps")):
        if durations:
            ordered = sorted(durations)
            print(f"  {title}: {len(durations)} calls, mean {sum(durations) / len(durations) / 1e3:.1f} us, "
                  f"p50 {ordered[len(ordered) // 2] / 1e3:.1f} us")
    for oname, t in covered_by.most_common(8):
        print(f"  covered by other threads' {oname}: {100 * t / total:.1f} % of the calls' time")


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+")
parser.add_argument("--call", default="cudaEventCreateWithFlags")
arguments = parser.parse_args()
for report in arguments.reports:
    label, _, path = report.partition("=")
    analyse(label, path, arguments.call)
