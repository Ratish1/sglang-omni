"""Which process holds a card while the thinker's submitted work waits on it, from an nsys trace of a server launched
under `baseline.py --nsys-session` (CUDA trace, graph level).

usage: python3 card0_contention.py TRACE.sqlite --serve-log SERVE.LOG [--device 0] [--skip-head-s 1] [--skip-tail-s 1]

Per process with device work on --device: its device busy time and share of the window and its heaviest kernels and
graphs, so a stage without a startup line is named by its kernels. For the thinker (its pid from the serve log's
sglang_ar_startup line): every device activity's wait from the start of its runtime call to its device start, the
union of those waits outside the thinker's own device work is time the thinker had work submitted and the card ran
something else or nothing; that union is split by the process whose device work covers it, the rest idle. Also each
process's device time inside the thinker's own device work, and the thinker's graph launch to start delay.
"""

import argparse
import re
import sqlite3
import statistics
from collections import defaultdict

DEVICE_TABLES = ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                 "CUPTI_ACTIVITY_KIND_MEMSET")
STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=\d+ .*pid=(\d+)")


def pid_of(global_id: int) -> int:
    # note (ratish): nsys packs the pid into 24 bits above the thread id; the bits above hold the host id
    return (global_id >> 24) & 0xFFFFFF


def merged(intervals: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def length(intervals: list[list[int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersect(first: list[list[int]], second: list[list[int]]) -> list[list[int]]:
    out = []
    i = j = 0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        if start < end:
            out.append([start, end])
        else:
            pass
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(first: list[list[int]], second: list[list[int]]) -> list[list[int]]:
    out = []
    j = 0
    for start, end in first:
        cursor = start
        while j < len(second) and second[j][1] <= cursor:
            j += 1
        k = j
        while k < len(second) and second[k][0] < end:
            if second[k][0] > cursor:
                out.append([cursor, second[k][0]])
            else:
                pass
            cursor = max(cursor, second[k][1])
            k += 1
        if cursor < end:
            out.append([cursor, end])
        else:
            pass
    return out


def clip(intervals: list[list[int]], start: int, end: int) -> list[list[int]]:
    return [[max(a, start), min(b, end)] for a, b in intervals if b > start and a < end]


def columns_of(db: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def device_rows(db: sqlite3.Connection, device: int) -> list[tuple[int, int, int, int, str]]:
    """(start, end, pid, correlation id, label) of every graph replay, kernel outside a graph, copy and memset."""
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    rows = []
    for table in DEVICE_TABLES:
        if table not in tables:
            continue
        else:
            pass
        columns = columns_of(db, table)
        if table.endswith("GRAPH_TRACE"):
            label = "graphExecId"
        elif table.endswith("KERNEL"):
            label = "shortName"
        else:
            label = None
        where = f"deviceId = {device}"
        if table.endswith("KERNEL") and "graphId" in columns:
            # note (ratish): a replayed graph's kernels carry the graph's correlation id; the graph row covers them
            where += " AND (graphId IS NULL OR graphId = 0)"
        else:
            pass
        query = f"SELECT start, end, globalPid, correlationId, {label or 'NULL'} FROM {table} WHERE {where}"
        kind = table.removeprefix("CUPTI_ACTIVITY_KIND_").lower()
        for start, end, global_pid, correlation, value in db.execute(query):
            if table.endswith("GRAPH_TRACE"):
                name = f"graph {value}"
            elif table.endswith("KERNEL"):
                name = strings.get(value, str(value))
            else:
                name = kind
            rows.append((start, end, pid_of(global_pid), correlation, name))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--serve-log", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip-head-s", type=float, default=1.0)
    parser.add_argument("--skip-tail-s", type=float, default=1.0)
    parser.add_argument("--stage", default="thinker", help="the stage whose waits are attributed, e.g. talker_ar")
    args = parser.parse_args()
    stages = {}
    with open(args.serve_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := STARTUP.search(line):
                stages[int(match[2])] = match[1]
            else:
                pass
    thinker = next(pid for pid, stage in stages.items() if stage == args.stage)
    db = sqlite3.connect(args.trace)
    rows = device_rows(db, args.device)
    # note (ratish): device records buffered since launch reach back before the collection; the runtime
    # calls cover only the collection, so they set the window
    first_call, last_call = db.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
    window_start = first_call + int(args.skip_head_s * 1e9)
    window_end = last_call - int(args.skip_tail_s * 1e9)
    window = window_end - window_start
    by_pid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    time_by_name: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for start, end, pid, _, name in rows:
        if end > window_start and start < window_end:
            by_pid[pid].append((start, end))
            time_by_name[pid][name] += min(end, window_end) - max(start, window_start)
        else:
            pass
    busy = {pid: clip(merged(intervals), window_start, window_end) for pid, intervals in by_pid.items()}
    every = merged([tuple(interval) for intervals in busy.values() for interval in intervals])
    print(f"device {args.device} window {window / 1e6:.0f} ms, any process busy {length(every) / 1e6:.0f} ms "
          f"({100 * length(every) / window:.1f} %), thinker pid {thinker}")
    print("| pid | stage | busy ms | busy % | heaviest work (ms) |")
    print("|---|---|---|---|---|")
    for pid, intervals in sorted(busy.items(), key=lambda item: -length(item[1])):
        heaviest = sorted(time_by_name[pid].items(), key=lambda item: -item[1])[:3]
        names = "; ".join(f"{name[:48]} {value / 1e6:.0f}" for name, value in heaviest)
        print(f"| {pid} | {stages.get(pid, '?')} | {length(intervals) / 1e6:.0f} | "
              f"{100 * length(intervals) / window:.1f} | {names} |")

    thinker_rows = [row for row in rows if row[2] == thinker and window_start <= row[0] and row[1] <= window_end]
    correlations = {row[3] for row in thinker_rows}
    calls = {}
    query = "SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE ((globalTid >> 24) & 16777215) = ?"
    for correlation, start in db.execute(query, (thinker,)):
        if correlation in correlations:
            calls[correlation] = start
        else:
            pass
    waits = merged([(calls[row[3]], row[0]) for row in thinker_rows if row[3] in calls and calls[row[3]] < row[0]])
    thinker_busy = busy[thinker]
    waiting = subtract(clip(waits, window_start, window_end), thinker_busy)
    graphs = [row for row in thinker_rows if row[4].startswith("graph ")]
    replays = max(len(graphs), 1)
    others = merged([tuple(interval) for pid, intervals in busy.items() if pid != thinker for interval in intervals])
    unmatched = sum(1 for row in thinker_rows if row[3] not in calls)
    print(f"\n{args.stage}: {len(thinker_rows)} device activities ({unmatched} without a runtime call), "
          f"{len(graphs)} graph replays, one replay every {window / replays / 1e6:.2f} ms, "
          f"device busy {length(thinker_busy) / replays / 1e6:.2f} ms a replay")
    print(f"submitted and waiting while the card runs no {args.stage} work: {length(waiting) / 1e6:.0f} ms "
          f"({length(waiting) / replays / 1e6:.3f} ms a replay)")
    print(f"| pid | stage | ms of that wait | ms a replay | ms inside the {args.stage}'s own device work |")
    print("|---|---|---|---|---|")
    for pid, intervals in sorted(busy.items(), key=lambda item: -length(intersect(waiting, item[1]))):
        if pid == thinker:
            continue
        else:
            pass
        covered = length(intersect(waiting, intervals))
        inside = length(intersect(thinker_busy, intervals))
        print(f"| {pid} | {stages.get(pid, '?')} | {covered / 1e6:.0f} | {covered / replays / 1e6:.3f} | "
              f"{inside / 1e6:.0f} |")
    idle = length(subtract(waiting, others))
    print(f"| | idle (no process busy) | {idle / 1e6:.0f} | {idle / replays / 1e6:.3f} | |")

    delays_clear, delays_covered = [], []
    for start, end, _, correlation, _ in graphs:
        if correlation not in calls:
            continue
        else:
            pass
        call = calls[correlation]
        hit = length(intersect([[call, start]], others)) if call < start else 0
        (delays_covered if hit > 0 else delays_clear).append((start - call) / 1e3)
    for name, delays in (("no other process busy", delays_clear), ("another process busy", delays_covered)):
        if delays:
            ordered = sorted(delays)
            print(f"graph launch to start, {name}: {len(ordered)} replays, median {statistics.median(ordered):.0f} us, "
                  f"p90 {ordered[int(0.9 * len(ordered))]:.0f} us, p99 {ordered[int(0.99 * len(ordered))]:.0f} us")
        else:
            print(f"graph launch to start, {name}: none")


if __name__ == "__main__":
    main()
