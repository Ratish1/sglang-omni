"""Talker decode steps of an nsys census split by what their prepare did.

A launch step with no host to device copy kept every row (the steady path); one with a
copy rebuilt or compacted rows. Synchronous steps are split the same way. Walls are the
NVTX range on the talker scheduler thread; api is CUDA runtime time inside it.

usage: python talker_step_classes.py serve.sqlite [serve.sqlite ...]
"""

import sqlite3
import statistics
import sys
from collections import defaultdict

KINDS = ("sched.launch decode", "sched.batch decode")


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


for path in sys.argv[1:]:
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    ranges = []
    for start, end, thread, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS where end is not null"
    ):
        label = text if text is not None else strings.get(text_id, "")
        name = " ".join(label.split(" ")[:2])
        if name in KINDS:
            ranges.append((start, end, thread, name))
    # note (ratish): only the talker launches decode steps ahead, so its thread is the one
    # with launch ranges; the thinker's synchronous steps stay out.
    threads = {thread for _, _, thread, name in ranges if name == KINDS[0]}
    ranges = [item for item in ranges if item[2] in threads]
    h2d_correlations = {
        (pid_of(global_pid), correlation)
        for correlation, global_pid in db.execute(
            "select correlationId, globalPid from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 1"
        )
    }
    calls = defaultdict(list)
    for start, end, thread, correlation in db.execute(
        "select start, end, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        if thread in threads:
            calls[thread].append((start, end, (pid_of(thread), correlation)))
        else:
            pass
    for thread in calls:
        calls[thread].sort()
    classes = defaultdict(lambda: {"wall": [], "api": [], "launches": []})
    for start, end, thread, name in ranges:
        inside = [call for call in calls[thread] if call[0] >= start and call[1] <= end]
        copies = sum(1 for call in inside if call[2] in h2d_correlations)
        label = f"{name} {'copy' if copies else 'none'}"
        classes[label]["wall"].append((end - start) / 1e6)
        classes[label]["api"].append(sum(call[1] - call[0] for call in inside) / 1e6)
        classes[label]["launches"].append(len(inside))
    print(f"== {path}")
    print(
        f"  {'class':32s} {'n':>5s} {'wall':>7s} {'p50':>7s} {'p90':>7s} {'api':>7s} {'calls':>6s} {'total':>8s}"
    )
    for label in sorted(classes):
        row = classes[label]
        print(
            f"  {label:32s} {len(row['wall']):5d} {statistics.fmean(row['wall']):7.2f}"
            f" {percentile(row['wall'], 0.5):7.2f} {percentile(row['wall'], 0.9):7.2f}"
            f" {statistics.fmean(row['api']):7.2f} {statistics.fmean(row['launches']):6.1f}"
            f" {sum(row['wall']):8.0f}"
        )
