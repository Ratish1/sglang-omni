"""How the talker's device work over each of its prefill passes sits against the thinker's on one card.

A pass runs from its forward's launch to the end of its token wait. Per pass: the talker's device busy and span (first
device start to last device end inside the pass), the idle gaps inside that span and the thinker's device time inside them.
Over the trace: the thinker's launch to device start lag, for launches inside a pass and outside every pass.

usage: python3 extend_bursts.py TRACE.sqlite SERVE.LOG
"""

import bisect
import re
import sqlite3
import statistics
import sys
from collections import defaultdict

STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=(\d+) .*pid=(\d+)")
DEVICE_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
)


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def merged(intervals: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def clipped(spans: list[list[int]], start: int, end: int) -> list[tuple[int, int]]:
    first = bisect.bisect_right([span[1] for span in spans], start)
    inside = []
    for span_start, span_end in spans[first:]:
        if span_start >= end:
            break
        inside.append((max(span_start, start), min(span_end, end)))
    return inside


def summary(values: list[float]) -> str:
    if not values:
        return "none"
    ordered = sorted(values)
    return (
        f"n {len(ordered)} mean {statistics.fmean(ordered):.3f} p50 {ordered[len(ordered) // 2]:.3f} "
        f"p90 {ordered[int(len(ordered) * 0.9)]:.3f} p99 {ordered[int(len(ordered) * 0.99)]:.3f}"
    )


def main() -> None:
    trace, serve_log = sys.argv[1:3]
    stages = {}
    with open(serve_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := STARTUP.search(line):
                stages[match[1]] = int(match[3])
    talker, thinker = stages["talker_ar"], stages["thinker"]

    db = sqlite3.connect(trace)
    names = {
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    session = db.execute(
        "SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()
    device: dict[int, list[tuple[int, int]]] = defaultdict(list)
    device_start: dict[tuple[int, int], int] = {}
    for table in DEVICE_TABLES:
        if table not in names:
            continue
        for correlation, global_pid, start, end in db.execute(
            f"SELECT correlationId, globalPid, start, end FROM {table} WHERE start >= ? AND end <= ? AND end >= start",
            session,
        ):
            pid = pid_of(global_pid)
            device[pid].append((start, end))
            key = (pid, correlation)
            device_start[key] = min(device_start.get(key, start), start)
    busy = {pid: merged(spans) for pid, spans in device.items()}

    text = "coalesce(text, (SELECT value FROM StringIds WHERE id = textId))"
    forwards, waits = [], []
    for start, end, label, global_tid in db.execute(
        f"SELECT start, end, {text}, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start"
    ):
        if pid_of(global_tid) != talker or not label:
            continue
        if label.startswith("omni.forward step[EXTEND"):
            forwards.append(start)
        elif label == "omni.host_tokens_wait":
            waits.append((start, end))
    forwards.sort()
    waits.sort()

    passes = []
    for forward in forwards:
        index = bisect.bisect_left(waits, (forward, 0))
        if index < len(waits):
            passes.append((forward, waits[index][1]))

    (
        talker_busy,
        talker_span,
        gap_total,
        gap_longest,
        thinker_in_gaps,
        thinker_in_pass,
    ) = ([], [], [], [], [], [])
    for start, end in passes:
        own = clipped(busy[talker], start, end)
        if not own:
            continue
        talker_busy.append(sum(b - a for a, b in own) / 1e6)
        talker_span.append((own[-1][1] - own[0][0]) / 1e6)
        gaps = [(a[1], b[0]) for a, b in zip(own, own[1:])]
        gap_total.append(sum(b - a for a, b in gaps) / 1e6)
        gap_longest.append(max((b - a for a, b in gaps), default=0) / 1e6)
        thinker_in_gaps.append(
            sum(b - a for gap in gaps for a, b in clipped(busy[thinker], *gap)) / 1e6
        )
        thinker_in_pass.append(
            sum(b - a for a, b in clipped(busy[thinker], start, end)) / 1e6
        )

    pass_starts = [start for start, _ in passes]
    inside, outside = [], []
    for start, name_id, global_tid, correlation in db.execute(
        "SELECT start, nameId, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        if pid_of(global_tid) != thinker or (thinker, correlation) not in device_start:
            continue
        lag = (device_start[(thinker, correlation)] - start) / 1e6
        index = bisect.bisect_right(pass_starts, start) - 1
        if index >= 0 and start < passes[index][1]:
            inside.append(lag)
        else:
            outside.append(lag)

    print(f"{trace}: {len(passes)} talker prefill passes, ms")
    print(f"  talker device busy in pass   {summary(talker_busy)}")
    print(f"  talker device span in pass   {summary(talker_span)}")
    print(f"  idle gaps inside the span    {summary(gap_total)}")
    print(f"  longest gap inside the span  {summary(gap_longest)}")
    print(f"  thinker device in the gaps   {summary(thinker_in_gaps)}")
    print(f"  thinker device in the pass   {summary(thinker_in_pass)}")
    print(f"  thinker launch lag in a pass {summary(inside)}")
    print(f"  thinker launch lag elsewhere {summary(outside)}")


if __name__ == "__main__":
    main()
