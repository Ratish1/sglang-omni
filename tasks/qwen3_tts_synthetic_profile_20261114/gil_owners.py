"""Who holds the interpreter lock during the talker steps of one nsys report (SQLite export,
collected with --trace=python-gil and --python-functions-trace): hold and wait time per
thread, and each thread's hold time inside each marked function, innermost mark first."""

import argparse
import bisect
import sqlite3
from collections import defaultdict

HOLDING = "Holding GIL"
WAITING = "Waiting for GIL"
# note (ratish): the kit probe labels talker steps "step[...]", the loop probe "loop.step ...".
STEP_PREFIX = ("step[", "loop.step")


def load_ranges(connection: sqlite3.Connection) -> list[tuple[int, int, str, str]]:
    query = """
        select e.start, e.end, coalesce(s.value, e.text, ''),
            ((e.globalTid >> 24) & 16777215) || ':' || coalesce(t.value, '')
        from NVTX_EVENTS e
        left join StringIds s on s.id = e.textId
        left join ThreadNames n on n.globalTid = e.globalTid
        left join StringIds t on t.id = n.nameId
        where e.end is not null and e.end > e.start
    """
    return [tuple(row) for row in connection.execute(query)]


def overlap(intervals: list[tuple[int, int]], starts: list[int], start: int, end: int) -> int:
    total = 0
    index = max(bisect.bisect_right(starts, start) - 1, 0)
    while index < len(intervals) and intervals[index][0] < end:
        low, high = intervals[index]
        total += max(0, min(high, end) - max(low, start))
        index += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    ranges = load_ranges(sqlite3.connect(args.sqlite))
    steps = [(start, end) for start, end, text, _ in ranges if text.startswith(STEP_PREFIX)]
    window_start = min(start for start, _ in steps)
    window_end = max(end for _, end in steps)
    window_ms = (window_end - window_start) / 1e6
    print(f"talker steps {len(steps)}, window {window_ms:.1f} ms")
    holds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    waits: dict[str, int] = defaultdict(int)
    marks: list[tuple[int, int, str, str]] = []
    for start, end, text, thread in ranges:
        low, high = max(start, window_start), min(end, window_end)
        if high <= low:
            continue
        elif text == HOLDING:
            holds[thread].append((low, high))
        elif text == WAITING:
            waits[thread] += high - low
        elif "." in text:
            marks.append((low, high, text, thread))
        else:
            pass
    print("\n## interpreter lock per thread, ms over the window")
    print("hold | share of window | wait | thread")
    for thread, intervals in sorted(holds.items(), key=lambda item: -sum(h - l for l, h in item[1])):
        held = sum(high - low for low, high in intervals) / 1e6
        print(f"{held:9.1f} | {held / window_ms:6.1%} | {waits[thread] / 1e6:9.1f} | {thread}")
    for intervals in holds.values():
        intervals.sort()
    starts = {thread: [low for low, _ in intervals] for thread, intervals in holds.items()}
    held_in: dict[tuple[str, str], int] = defaultdict(int)
    calls: dict[tuple[str, str], int] = defaultdict(int)
    for low, high, text, thread in marks:
        if thread in holds:
            held_in[(thread, text)] += overlap(holds[thread], starts[thread], low, high)
            calls[(thread, text)] += 1
        else:
            pass
    print("\n## lock held inside each marked function (nested marks count in each), ms")
    print("held | calls | per call us | thread | function")
    for (thread, text), held in sorted(held_in.items(), key=lambda item: -item[1])[: args.top]:
        count = calls[(thread, text)]
        print(f"{held / 1e6:8.1f} | {count:6d} | {held / 1e3 / count:9.1f} | {thread} | {text}")


if __name__ == "__main__":
    main()
