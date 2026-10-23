"""Talker decode steps of an nsys census split by what their prepare did.

A step whose talker reused its decode buffers (probe mark "mr.decode_reuse hit=1") kept every
row in place; one that rebuilt them (hit=0) changed rows, and a step with neither mark predates
the mark. Each is split again by whether a host to device copy ran inside it. Walls are the NVTX
range on the talker scheduler thread, found by the probe's thread name mark; api is CUDA runtime
time inside it, rows the mean batch rows. With --window, only steps starting inside the
measured window count.

usage: python talker_step_classes.py serve.sqlite [serve.sqlite ...] [--window window.txt]
"""

import argparse
import bisect
import re
import sqlite3
import statistics
from collections import defaultdict

from nsys_stage_ledger import session_window

KINDS = ("sched.launch decode", "sched.batch decode")
MARK = 34
TALKER_THREAD = "thread name=scheduler-talker_ar "


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+")
parser.add_argument("--window")
args = parser.parse_args()
for path in args.reports:
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    threads = set()
    reuse_marks = defaultdict(list)
    for start, thread, text, text_id in db.execute(
        "select start, globalTid, text, textId from NVTX_EVENTS where eventType = ?",
        (MARK,),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith(TALKER_THREAD):
            threads.add(thread)
        elif label.startswith("mr.decode_reuse "):
            match = re.match(r"mr\.decode_reuse hit=(\d)", label)
            reuse_marks[thread].append((start, match.group(1) == "1"))
        else:
            pass
    for thread in reuse_marks:
        reuse_marks[thread].sort()
    ranges = []
    for start, end, thread, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS where end is not null"
    ):
        if thread not in threads or not t0 <= start <= t1:
            continue
        else:
            pass
        label = text if text is not None else strings.get(text_id, "")
        words = label.split(" ")
        name = " ".join(words[:2])
        if name in KINDS:
            rows = (
                int(words[2][3:])
                if len(words) > 2 and words[2].startswith("bs=")
                else 0
            )
            ranges.append((start, end, thread, name, rows))
        else:
            pass
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
    classes = defaultdict(lambda: {"wall": [], "api": [], "launches": [], "rows": []})
    for start, end, thread, name, rows in ranges:
        thread_calls = calls[thread]
        first = bisect.bisect_left(thread_calls, (start,))
        inside = []
        for call in thread_calls[first:]:
            if call[0] > end:
                break
            elif call[1] <= end:
                inside.append(call)
            else:
                pass
        copies = sum(1 for call in inside if call[2] in h2d_correlations)
        marks = reuse_marks.get(thread, [])
        position = bisect.bisect_left(marks, (start, False))
        if position < len(marks) and marks[position][0] <= end:
            reuse = "reuse" if marks[position][1] else "rebuild"
        else:
            reuse = "unmarked"
        label = f"{name} {reuse} {'copy' if copies else 'none'}"
        classes[label]["wall"].append((end - start) / 1e6)
        classes[label]["api"].append(sum(call[1] - call[0] for call in inside) / 1e6)
        classes[label]["launches"].append(len(inside))
        classes[label]["rows"].append(rows)
    print(f"== {path}")
    print(
        f"  {'class':40s} {'n':>5s} {'rows':>6s} {'wall':>7s} {'p50':>7s} {'p90':>7s} {'api':>7s} {'calls':>6s} {'total':>8s}"
    )
    for label in sorted(classes):
        row = classes[label]
        print(
            f"  {label:40s} {len(row['wall']):5d} {statistics.fmean(row['rows']):6.1f}"
            f" {statistics.fmean(row['wall']):7.2f}"
            f" {percentile(row['wall'], 0.5):7.2f} {percentile(row['wall'], 0.9):7.2f}"
            f" {statistics.fmean(row['api']):7.2f} {statistics.fmean(row['launches']):6.1f}"
            f" {sum(row['wall']):8.0f}"
        )
