"""The host waits a talker admission costs, from one probed nsys report (run on the box).

Every talker admission resolves the in-flight decode step first (the drain) and then runs its extend
synchronously. Per admission this reads the wall of the talker's resolve range right before it, and the time
inside the extend range spent in blocking CUDA calls (event and stream synchronizes, and copies that ran for
longer than the threshold, which are pageable or wait on the stream). Their sum is what an asynchronous admission
could give back to the talker's loop (design 05 section 9).

usage: python talker_admission_waits.py REPORT.sqlite [--window window.txt] [--copy-wait-us 50]
"""

from __future__ import annotations

import argparse
import bisect
import re
import sqlite3
import statistics

from nsys_stage_ledger import pid_of, session_window

MARK = 34
STEP = re.compile(r"^sched\.(batch|launch|resolve) (extend|decode|mixed)")
BLOCKING = ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    parser.add_argument("--copy-wait-us", type=float, default=50.0)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    talker_pids = set()
    steps = []
    for start, end, tid, text, text_id, event_type in db.execute(
        "select start, end, globalTid, text, textId, eventType from NVTX_EVENTS order by start"
    ):
        label = text if text is not None else strings.get(text_id, "")
        if not label:
            continue
        if event_type == MARK and label == "proc stage=talker_ar":
            talker_pids.add(pid_of(tid))
        match = STEP.match(label)
        if match and end is not None and t0 <= start <= t1:
            steps.append((start, end, tid, match.group(1), match.group(2)))
    steps = [s for s in steps if pid_of(s[2]) in talker_pids]
    threads = {s[2] for s in steps}
    calls: dict[int, list[tuple[int, int, str]]] = {thread: [] for thread in threads}
    for start, end, tid, name_id in db.execute(
        "select start, end, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start between ? and ?",
        (t0, t1),
    ):
        if tid in calls:
            calls[tid].append((start, end, strings.get(name_id, "")))
    for thread in calls:
        calls[thread].sort()
    starts = {thread: [c[0] for c in calls[thread]] for thread in calls}
    window_ms = (t1 - t0) / 1e6
    drains, blocks = [], []
    previous = None
    for step in steps:
        start, end, tid, phase, kind = step
        if kind in ("extend", "mixed") and phase == "batch":
            drain = 0.0
            if previous is not None and previous[3] == "resolve":
                drain = (previous[1] - previous[0]) / 1e6
            blocked = 0.0
            index = bisect.bisect_left(starts[tid], start)
            while index < len(calls[tid]) and calls[tid][index][0] < end:
                call_start, call_end, name = calls[tid][index]
                duration_us = (call_end - call_start) / 1e3
                if name.startswith(BLOCKING) or (
                    name.startswith("cudaMemcpy") and duration_us > args.copy_wait_us
                ):
                    blocked += duration_us / 1e3
                index += 1
            drains.append(drain)
            blocks.append(blocked)
        previous = step
    print(f"window {window_ms / 1e3:.2f} s, talker admissions {len(drains)}")
    if not drains:
        return
    total = sum(drains) + sum(blocks)
    print(
        f"drain before an admission: mean {statistics.fmean(drains):.2f} ms "
        f"({sum(1 for d in drains if d > 0)} admissions had one); "
        f"blocking calls inside it: mean {statistics.fmean(blocks):.2f} ms; "
        f"together {total:.0f} ms, {100 * total / window_ms:.1f} % of the window"
    )


if __name__ == "__main__":
    main()
