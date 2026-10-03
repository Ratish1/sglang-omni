"""Eager kernels and copies issued inside one kind of NVTX range, by the innermost child range
that issued them: count per range, and what each launch or copy was.

One sweep per thread: calls and the outer ranges are both sorted by start, so each call is
matched to its outer range without scanning; the innermost child comes from the ranges that
start inside that outer range.

usage: python nsys_range_launches.py serve.sqlite "sched.launch decode"
"""

import bisect
import sqlite3
import sys
from collections import defaultdict


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def kind_of(label: str) -> str:
    return " ".join(label.split(" ")[:2])


path, kind = sys.argv[1], sys.argv[2]
db = sqlite3.connect(path)
strings = dict(db.execute("select id, value from StringIds"))
ranges_by_thread = defaultdict(list)
for start, end, thread, text, text_id in db.execute(
    "select start, end, globalTid, text, textId from NVTX_EVENTS where end is not null and eventType = 59"
):
    label = text if text is not None else strings.get(text_id, "")
    ranges_by_thread[thread].append((start, end, label))
outer_by_thread = {}
for thread, items in ranges_by_thread.items():
    outer = sorted(
        (start, end) for start, end, label in items if kind_of(label) == kind
    )
    if outer:
        outer_by_thread[thread] = outer
    else:
        pass
outer_count = sum(len(outer) for outer in outer_by_thread.values())
calls_by_key = {}
for thread, outer in outer_by_thread.items():
    outer_starts = [start for start, _ in outer]
    children = sorted(
        (start, end, label)
        for start, end, label in ranges_by_thread[thread]
        if kind_of(label) != kind
    )
    child_starts = [start for start, _, _ in children]
    for start, end, correlation, name_id in db.execute(
        "select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ? order by start",
        (thread,),
    ):
        call = strings.get(name_id, "")
        if not (
            call.startswith("cudaLaunchKernel") or call.startswith("cudaMemcpyAsync")
        ):
            continue
        else:
            pass
        index = bisect.bisect_right(outer_starts, start) - 1
        if index < 0 or end > outer[index][1]:
            continue
        else:
            pass
        outer_start, outer_end = outer[index]
        # innermost child: the latest-starting child range inside the outer range that covers the call
        child_index = bisect.bisect_right(child_starts, start) - 1
        innermost = "(direct)"
        while child_index >= 0 and child_starts[child_index] >= outer_start:
            child_start, child_end, label = children[child_index]
            if child_end >= end:
                innermost = kind_of(label)
                break
            else:
                child_index -= 1
        calls_by_key[(pid_of(thread), correlation)] = (innermost, call)
kernel_names = {}
for correlation, global_pid, name_id in db.execute(
    "select correlationId, globalPid, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"
):
    key = (pid_of(global_pid), correlation)
    if key in calls_by_key:
        kernel_names[key] = strings.get(name_id, str(name_id))
    else:
        pass
copies = {}
for correlation, global_pid, copy_kind, size in db.execute(
    "select correlationId, globalPid, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"
):
    key = (pid_of(global_pid), correlation)
    if key in calls_by_key:
        copies[key] = f"memcpy kind {copy_kind} {size} B"
    else:
        pass
counts = defaultdict(int)
for key, (innermost, call) in calls_by_key.items():
    what = kernel_names.get(key) or copies.get(key) or call
    counts[(innermost, what[:110])] += 1
print(f"== {path}: {outer_count} x '{kind}'")
for (child, what), count in sorted(counts.items(), key=lambda item: -item[1])[:40]:
    print(f"  {count / max(outer_count, 1):6.2f}/range  {child:24s} {what}")
