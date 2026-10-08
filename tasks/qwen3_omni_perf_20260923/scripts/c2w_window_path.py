"""Where a code2wav window's delay goes, from a probed census: its inbox, code2wav itself, and the hops after it.

A request's window-completing codec chunk (its stream chunk number 9 completes the first window, every later one the
next, as in c2w_batch_opportunity.py) is put in code2wav's inbox ("q code2wav.in.put ... t=stream_chunk") and taken
("... .get"); the window's audio is put in code2wav's outbox ("q code2wav.out.put"), the k-th audio put for the k-th
window, and reaches the coordinator ("coord.recv ... from=code2wav n>0"). Splits each window's delay from the talker's
message to the coordinator into inbox (put to get), code2wav (get to the audio's put: waiting among the ready windows,
the replay, the copy to the host) and after (outbox to the coordinator), by percentile and by band of the whole delay.
Then each band's inbox wait by what code2wav's scheduler thread did meanwhile, in this priority: a CUDA synchronize
call, a wait for the GIL (python-gil trace), a c2w range of the probe, anything else (its own Python outside the probe's
ranges, or sleeping on a launched window).

usage: python3 c2w_window_path.py REPORT.sqlite --window window.txt
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
FIRST_WINDOW_CHUNK = 9


def merged(intervals: list[tuple[int, int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def covered(spans: list[list[int]], starts: list[int], a: int, b: int) -> int:
    """Length of [a, b] that a merged, sorted interval list covers."""
    total = 0
    index = max(bisect.bisect_right(starts, a) - 1, 0)
    while index < len(spans) and spans[index][0] < b:
        total += max(0, min(b, spans[index][1]) - max(a, spans[index][0]))
        index += 1
    return total


def pct(values: list[int], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    puts: dict[str, list[int]] = collections.defaultdict(list)
    gets: dict[str, list[int]] = collections.defaultdict(list)
    outs: dict[str, list[int]] = collections.defaultdict(list)
    arrived: dict[str, list[int]] = collections.defaultdict(list)
    out_types: collections.Counter = collections.Counter()
    names: dict[int, str] = {}
    thread_ranges: dict[int, list] = collections.defaultdict(list)
    for start, end, tid, text, text_id, kind in db.execute(
        "select start, end, globalTid, text, textId, eventType from NVTX_EVENTS where eventType in (?, ?) "
        "order by start",
        (PUSH_POP, MARK),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if kind == PUSH_POP:
            if end is not None and (
                label.startswith("Waiting for GIL") or label.startswith("c2w.")
            ):
                thread_ranges[tid].append((start, end, label))
            else:
                pass
            continue
        else:
            pass
        if label.startswith("thread name="):
            names[tid] = label.split(" ")[1][5:]
            continue
        else:
            pass
        match = re.match(r"q code2wav\.in\.(put|get) rid=(\S+) t=stream_chunk", label)
        if match:
            (puts if match.group(1) == "put" else gets)[match.group(2)].append(start)
            continue
        else:
            pass
        match = re.match(r"q code2wav\.out\.put rid=(\S+) t=(\S+)", label)
        if match:
            out_types[match.group(2)] += 1
            if match.group(2) == "stream":
                outs[match.group(1)].append(start)
            else:
                pass
            continue
        else:
            pass
        match = re.match(r"coord\.recv rid=(\S+) from=code2wav n=(\d+)", label)
        if match and int(match.group(2)) > 0:
            arrived[match.group(1)].append(start)
        else:
            pass
    print(
        f"window {(t1 - t0) / 1e9:.1f} s; code2wav outbox puts by type: {dict(out_types)}"
    )
    rows = []
    for request_id, request_puts in puts.items():
        request_gets, request_outs = gets.get(request_id, []), outs.get(request_id, [])
        request_arrived = arrived.get(request_id, [])
        windows = len(request_puts) - FIRST_WINDOW_CHUNK
        for k in range(min(windows, len(request_outs), len(request_arrived))):
            put, get = (
                request_puts[FIRST_WINDOW_CHUNK + k],
                request_gets[FIRST_WINDOW_CHUNK + k],
            )
            out, coord = request_outs[k], request_arrived[k]
            if t0 <= coord <= t1 and put <= get <= out <= coord:
                rows.append((coord - put, get - put, out - get, coord - out, put, get))
            else:
                pass
    if not rows:
        print("no windows matched; check the probe marks")
        return
    else:
        pass
    print(f"  {len(rows)} windows; ms        {'p50':>8}{'p90':>8}{'p99':>8}{'mean':>8}")
    for index, name in enumerate(("whole delay", "inbox", "code2wav", "after")):
        values = [row[index] for row in rows]
        print(
            f"  {name:<20}{pct(values, .5):>8.1f}{pct(values, .9):>8.1f}{pct(values, .99):>8.1f}"
            f"{statistics.fmean(values) / 1e6:>8.1f}"
        )
    rows.sort()
    n = len(rows)
    print("  by band of the whole delay, mean ms: whole, inbox, code2wav, after")
    for name, lo, hi in (
        ("<= p50", 0, n // 2),
        ("p50 to p90", n // 2, int(n * 0.9)),
        ("p90 to p99", int(n * 0.9), int(n * 0.99)),
        ("> p99", int(n * 0.99), n),
    ):
        band = rows[lo:hi]
        if not band:
            continue
        else:
            pass
        print(
            f"  {name:<12}{len(band):>6}"
            + "".join(
                f"{statistics.fmean(row[i] for row in band) / 1e6:>9.1f}"
                for i in range(4)
            )
        )
    code2wav = next(
        (tid for tid, name in names.items() if name == "scheduler-code2wav"), None
    )
    if code2wav is None:
        print("no code2wav scheduler thread name mark")
        return
    else:
        pass
    sync = [
        (start, end)
        for start, end, name_id in db.execute(
            "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ?",
            (code2wav,),
        )
        if "Synchronize" in strings.get(name_id, "")
    ]
    gil = [
        (a, b)
        for a, b, label in thread_ranges[code2wav]
        if label.startswith("Waiting for GIL")
    ]
    ranged = [
        (a, b) for a, b, label in thread_ranges[code2wav] if label.startswith("c2w.")
    ]
    layers = [merged(sync), merged(sync + gil), merged(sync + gil + ranged)]
    layer_starts = [[span[0] for span in layer] for layer in layers]
    print(
        "  inbox wait by what code2wav's thread did meanwhile, mean ms: sync, gil, c2w range, else"
    )
    for name, lo, hi in (
        ("<= p50", 0, n // 2),
        ("p50 to p90", n // 2, int(n * 0.9)),
        ("p90 to p99", int(n * 0.9), int(n * 0.99)),
        ("> p99", int(n * 0.99), n),
    ):
        band = rows[lo:hi]
        if not band:
            continue
        else:
            pass
        parts = [0.0, 0.0, 0.0, 0.0]
        for row in band:
            a, b = row[4], row[5]
            levels = [
                covered(layer, starts, a, b)
                for layer, starts in zip(layers, layer_starts)
            ]
            parts[0] += levels[0]
            parts[1] += levels[1] - levels[0]
            parts[2] += levels[2] - levels[1]
            parts[3] += (b - a) - levels[2]
        print(
            f"  {name:<12}{len(band):>6}"
            + "".join(f"{value / len(band) / 1e6:>9.1f}" for value in parts)
        )


if __name__ == "__main__":
    main()
