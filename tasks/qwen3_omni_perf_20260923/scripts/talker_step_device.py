"""Host wall against device time of each talker decode step, by the step's rows.

A synchronous talker decode step ("sched.batch decode bs=N" on the talker scheduler thread) ends
after it waits for its own kernels, so its device time is the union of the talker process's
kernel intervals inside the range. Steps inside the measured window only; with node graph
tracing every replayed kernel is its own interval.

usage: python3 talker_step_device.py REPORT.sqlite --window window.txt
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics

from nsys_stage_ledger import session_window

PUSH_POP = 59
MARK = 34
STEP = re.compile(r"sched\.batch decode bs=(\d+)")
BANDS = (1, 9, 17, 25, 33, 41, 49, 57)


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def union_inside(
    intervals: list[tuple[int, int]], starts: list[int], a: int, b: int
) -> tuple[int, int]:
    """Busy time of the sorted intervals clipped to [a, b], and how many start inside it."""
    index = max(bisect.bisect_left(starts, a) - 1, 0)
    busy = 0
    count = 0
    covered = a
    while index < len(intervals) and intervals[index][0] < b:
        start, end = intervals[index]
        index += 1
        if end <= covered:
            continue
        else:
            pass
        if start >= a:
            count += 1
        else:
            pass
        busy += min(end, b) - max(start, covered)
        covered = min(max(covered, end), b)
    return busy, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    talker = None
    for thread, text, text_id in db.execute(
        "select globalTid, text, textId from NVTX_EVENTS where eventType = ?", (MARK,)
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith("thread name=scheduler-talker_ar "):
            talker = thread
        else:
            pass
    if talker is None:
        print("no talker scheduler thread")
        return
    else:
        pass
    talker_pid = pid_of(talker)
    kernels = sorted(
        (start, end)
        for start, end, global_pid in db.execute(
            "select start, end, globalPid from CUPTI_ACTIVITY_KIND_KERNEL where start between ? and ?",
            (t0 - 1_000_000_000, t1 + 1_000_000_000),
        )
        if pid_of(global_pid) == talker_pid
    )
    starts = [start for start, _ in kernels]
    bands: dict[int, list[tuple[float, float, int]]] = collections.defaultdict(list)
    for start, end, text, text_id in db.execute(
        "select start, end, text, textId from NVTX_EVENTS where globalTid = ? and eventType = ? "
        "and end is not null and start between ? and ?",
        (talker, PUSH_POP, t0, t1),
    ):
        label = text if text is not None else strings.get(text_id, "")
        match = STEP.match(label)
        if match is None:
            continue
        else:
            pass
        rows = int(match.group(1))
        busy, count = union_inside(kernels, starts, start, end)
        band = BANDS[bisect.bisect_right(BANDS, rows) - 1]
        bands[band].append(((end - start) / 1e6, busy / 1e6, count))
    print(
        "talker decode steps by rows: host wall, device busy inside the step, kernels started (means)"
    )
    print(f"  {'rows':<8}{'steps':>7}{'wall':>8}{'device':>9}{'idle':>8}{'kernels':>9}")
    for band in BANDS:
        steps = bands.get(band)
        if not steps:
            continue
        else:
            pass
        wall = statistics.fmean(step[0] for step in steps)
        device = statistics.fmean(step[1] for step in steps)
        print(
            f"  {band:>2}-{band + 7:<5}{len(steps):>7}{wall:>8.2f}{device:>9.2f}{wall - device:>8.2f}"
            f"{statistics.fmean(step[2] for step in steps):>9.0f}"
        )


if __name__ == "__main__":
    main()
