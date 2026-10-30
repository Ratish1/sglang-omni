"""Who holds the GIL while an engine's scheduler thread waits for it, and what the holder was doing.

For every GIL wait (python-gil trace) of a thread named scheduler-<stage> inside the window,
the time it overlaps a GIL hold of another thread of the same process is charged to that
holder thread and to the holder's activity at that moment, sampled every 10 us: the holder's
innermost probe range (io.ser, io.deser, sched.*, c2w.*, ...), else its innermost open stage
IO bracket (io.chunk, io.recv, io.coord, io.stream, io.send, io.result, built from their
entry and .end marks), else "unranged" (control plane decode, asyncio and anything the probe
does not name). Prints per waiting thread its total wait and the charged share, then per
holder thread and activity the charged ms and the share of the wait.

usage: python gil_holder_activity.py REPORT.sqlite --window window.txt
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3

from nsys_stage_ledger import pid_of, session_window
from omni_census import MARK, PUSH_POP, kind_of

STEP_NS = 10_000
BRACKET = re.compile(r"^(io\.\w+)( rid=.*)$")
BRACKET_END = re.compile(r"^(io\.\w+)\.end( rid=.*)$")


def innermost(
    intervals: list[tuple[int, int, str]], starts: list[int], t: int
) -> str | None:
    """The latest starting interval that still covers t (nested ranges or open brackets)."""
    index = bisect.bisect_right(starts, t) - 1
    scanned = 0
    while index >= 0 and scanned < 256:
        start, end, label = intervals[index]
        if end > t:
            return label
        index -= 1
        scanned += 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    names: dict[int, str] = {}
    waits: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    holds: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    ranges: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    open_brackets: dict[tuple[int, str], list[int]] = collections.defaultdict(list)
    brackets: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    for start, end, tid, text, text_id, event_type in db.execute(
        "select start, end, globalTid, text, textId, eventType from NVTX_EVENTS "
        "where eventType in (?, ?) order by start",
        (PUSH_POP, MARK),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if not label:
            continue
        else:
            pass
        if event_type == MARK:
            if label.startswith("thread name="):
                names[tid] = label.split(" ")[1][5:]
                continue
            else:
                pass
            match = BRACKET_END.match(label)
            if match is not None:
                key = (tid, match.group(1) + match.group(2))
                if open_brackets[key]:
                    entry = open_brackets[key].pop(0)
                    # the activity keeps the kind and its stages, not the request or a run's count
                    words = [
                        word
                        for word in match.group(2).split()
                        if not word.startswith("rid=") and not word.startswith("n=")
                    ]
                    brackets[tid].append(
                        (entry, start, " ".join([match.group(1), *words]))
                    )
                else:
                    pass
                continue
            else:
                pass
            match = BRACKET.match(label)
            if match is not None:
                open_brackets[(tid, match.group(1) + match.group(2))].append(start)
            else:
                pass
            continue
        else:
            pass
        if end is None or end <= t0 or start >= t1:
            continue
        else:
            pass
        if label.startswith("Waiting for GIL"):
            waits[tid].append((max(start, t0), min(end, t1)))
        elif label.startswith("Holding GIL"):
            holds[tid].append((start, end))
        else:
            ranges[tid].append((start, end, kind_of(label)))
    for table in (ranges, brackets):
        for items in table.values():
            items.sort()
    range_starts = {tid: [item[0] for item in items] for tid, items in ranges.items()}
    bracket_starts = {
        tid: [item[0] for item in items] for tid, items in brackets.items()
    }
    for tid in holds:
        holds[tid].sort()
    hold_starts = {tid: [item[0] for item in items] for tid, items in holds.items()}
    threads_of_pid: dict[int, list[int]] = collections.defaultdict(list)
    for tid in set(holds) | set(waits):
        threads_of_pid[pid_of(tid)].append(tid)
    for waiter in sorted(waits, key=lambda tid: names.get(tid, "")):
        name = names.get(waiter, "")
        if not name.startswith("scheduler-"):
            continue
        else:
            pass
        total_wait = sum(end - start for start, end in waits[waiter])
        charged: collections.Counter = collections.Counter()
        for start, end in waits[waiter]:
            for holder in threads_of_pid[pid_of(waiter)]:
                if holder == waiter or holder not in holds:
                    continue
                else:
                    pass
                index = max(bisect.bisect_right(hold_starts[holder], start) - 1, 0)
                while index < len(holds[holder]) and holds[holder][index][0] < end:
                    hold_start, hold_end = holds[holder][index]
                    a, b = max(start, hold_start), min(end, hold_end)
                    t = a
                    while t < b:
                        weight = min(STEP_NS, b - t)
                        activity = None
                        if holder in ranges:
                            activity = innermost(
                                ranges[holder], range_starts[holder], t
                            )
                        else:
                            pass
                        if activity is None and holder in brackets:
                            activity = innermost(
                                brackets[holder], bracket_starts[holder], t
                            )
                        else:
                            pass
                        charged[
                            (names.get(holder, str(holder)), activity or "unranged")
                        ] += weight
                        t += weight
                    index += 1
        charged_total = sum(charged.values())
        print(
            f"== {name} (pid {pid_of(waiter)}): GIL wait {total_wait / 1e6:.1f} ms in the window,"
            f" {charged_total / 1e6:.1f} ms of it overlapping another thread's hold"
        )
        for (holder_name, activity), ns in charged.most_common(25):
            print(
                f"  {holder_name:<28} {activity:<44} {ns / 1e6:8.1f} ms {100 * ns / max(total_wait, 1):5.1f} %"
            )


if __name__ == "__main__":
    main()
