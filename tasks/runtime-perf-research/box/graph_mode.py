"""Talker graph replay durations by what ran on the card just before them.

A context's graph replays slow down once a second stream has issued eight eager launches, and only another process's context
running clears that. Per talker graph: replays and duration split by whether the thinker's device work ran between the
talker's previous device work and this replay, and how many of the talker's eager launches on a stream other than the
replay's own came since the thinker last ran.

usage: python3 graph_mode.py TRACE.sqlite SERVE.LOG [--top 12]
"""

import argparse
import bisect
import re
import sqlite3
import statistics
from collections import defaultdict

STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=(\d+) .*pid=(\d+)")
EAGER_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMSET")
SLOW_MODE_LAUNCHES = 8


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


def describe(values: list[float]) -> str:
    if not values:
        return "n     0"
    ordered = sorted(values)
    return (
        f"n {len(ordered):5d} mean {statistics.fmean(ordered) / 1e3:7.3f} "
        f"p50 {ordered[len(ordered) // 2] / 1e3:7.3f} total {sum(ordered) / 1e6:8.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("serve_log")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    stages = {}
    with open(args.serve_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := STARTUP.search(line):
                stages[match[1]] = int(match[3])
    talker, thinker = stages["talker_ar"], stages["thinker"]

    db = sqlite3.connect(args.trace)
    names = {
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    session = db.execute(
        "SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()

    graphs = []
    other_busy = []
    talker_device = []
    for global_pid, graph_id, stream, start, end in db.execute(
        "SELECT globalPid, graphId, streamId, start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
        "WHERE start >= ? AND end <= ? AND end >= start",
        session,
    ):
        if pid_of(global_pid) == talker:
            graphs.append((start, end, graph_id, stream))
            talker_device.append((start, end))
        elif pid_of(global_pid) == thinker:
            other_busy.append((start, end))
    eager = []
    for table in EAGER_TABLES:
        if table not in names:
            continue
        for global_pid, stream, start, end in db.execute(
            f"SELECT globalPid, streamId, start, end FROM {table} WHERE start >= ? AND end <= ? AND end >= start",
            session,
        ):
            if pid_of(global_pid) == talker:
                eager.append((start, stream))
                talker_device.append((start, end))
            elif pid_of(global_pid) == thinker:
                other_busy.append((start, end))
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in names:
        for global_pid, start, end in db.execute(
            "SELECT globalPid, start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ? AND end >= start",
            session,
        ):
            if pid_of(global_pid) == talker:
                talker_device.append((start, end))
            elif pid_of(global_pid) == thinker:
                other_busy.append((start, end))

    graphs.sort()
    eager.sort()
    talker_spans = merged(talker_device)
    talker_ends = [span[1] for span in talker_spans]
    other_spans = merged(other_busy)
    other_starts = [span[0] for span in other_spans]
    other_ends = [span[1] for span in other_spans]
    eager_starts = [start for start, _ in eager]
    streams = sorted({stream for _, _, _, stream in graphs})
    eager_by_stream = defaultdict(int)
    for _, stream in eager:
        eager_by_stream[stream] += 1
    print(
        f"{args.trace}: talker graph replays {len(graphs)}, on streams {streams}; eager launches by stream {dict(eager_by_stream)}"
    )

    by_graph = defaultdict(lambda: defaultdict(list))
    for start, end, graph_id, stream in graphs:
        previous = bisect.bisect_left(talker_ends, start) - 1
        previous_end = talker_ends[previous] if previous >= 0 else session[0]
        other = bisect.bisect_left(other_starts, previous_end)
        ran_between = other < len(other_starts) and other_starts[other] < start
        last_other = bisect.bisect_left(other_ends, start) - 1
        last_other_end = other_ends[last_other] if last_other >= 0 else session[0]
        low, high = bisect.bisect_left(
            eager_starts, last_other_end
        ), bisect.bisect_left(eager_starts, start)
        side_stream_eager = sum(
            1 for _, eager_stream in eager[low:high] if eager_stream != stream
        )
        duration = end - start
        by_graph[graph_id]["all"].append(duration)
        by_graph[graph_id][
            "thinker just ran" if ran_between else "talker continued"
        ].append(duration)
        by_graph[graph_id][
            (
                "side eager >= 8"
                if side_stream_eager >= SLOW_MODE_LAUNCHES
                else "side eager < 8"
            )
        ].append(duration)

    top = sorted(by_graph.items(), key=lambda item: -sum(item[1]["all"]))[: args.top]
    for graph_id, rows in top:
        print(f"  graph {graph_id}")
        for name in (
            "all",
            "thinker just ran",
            "talker continued",
            "side eager < 8",
            "side eager >= 8",
        ):
            print(f"    {name:18s} {describe(rows[name])}")


if __name__ == "__main__":
    main()
