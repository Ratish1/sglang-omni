"""Host wall against device time of each talker decode step, by the step's rows.

A synchronous talker decode step ("sched.batch decode bs=N" on the talker scheduler thread) ends
after it waits for its own kernels, so its device time is the union of the talker process's
kernel intervals inside the range. Steps inside the measured window only; with node graph
tracing every replayed kernel is its own interval. --names compares two row bands' kernels per
step by name.

usage: python3 talker_step_device.py REPORT.sqlite --window window.txt [--names 25,57]
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
    parser.add_argument(
        "--names", help="two row bands to compare by kernel name, e.g. 25,57"
    )
    parser.add_argument("--top", type=int, default=25)
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
    named = sorted(
        (start, end, name_id)
        for start, end, global_pid, name_id in db.execute(
            "select start, end, globalPid, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start between ? and ?",
            (t0 - 1_000_000_000, t1 + 1_000_000_000),
        )
        if pid_of(global_pid) == talker_pid
    )
    kernels = [(start, end) for start, end, _ in named]
    starts = [start for start, _ in kernels]
    compared = {int(band) for band in args.names.split(",")} if args.names else set()
    names: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    durations: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
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
        if band in compared:
            index = bisect.bisect_left(starts, start)
            while index < len(named) and named[index][0] < end:
                kernel_start, kernel_end, name_id = named[index]
                names[band][strings.get(name_id, "?")] += 1
                durations[band][strings.get(name_id, "?")] += kernel_end - kernel_start
                index += 1
        else:
            pass
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
    if len(compared) == 2:
        low, high = sorted(compared)
        print(
            f"kernels per step, rows {low}-{low + 7} against {high}-{high + 7} (count, device us), by change"
        )
        per_step = {
            band: {
                name: (
                    names[band][name] / len(bands[band]),
                    durations[band][name] / len(bands[band]) / 1e3,
                )
                for name in names[band]
            }
            for band in (low, high)
            if bands.get(band)
        }
        if len(per_step) == 2:
            every = set(per_step[low]) | set(per_step[high])
            changes = sorted(
                every,
                key=lambda name: -abs(
                    per_step[high].get(name, (0, 0))[1]
                    - per_step[low].get(name, (0, 0))[1]
                ),
            )
            for name in changes[: args.top]:
                count_low, us_low = per_step[low].get(name, (0, 0))
                count_high, us_high = per_step[high].get(name, (0, 0))
                print(
                    f"  {count_low:7.1f} {us_low:8.1f}  {count_high:7.1f} {us_high:8.1f}  {us_high - us_low:+8.1f}  {name[:90]}"
                )
        else:
            pass
    else:
        pass


if __name__ == "__main__":
    main()
