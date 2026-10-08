"""Where a code2wav window's delay goes, from a probed census: its inbox, code2wav itself, and the hops after it.

A request's window-completing codec chunk (its stream chunk number 9 completes the first window, every later one the
next, as in c2w_batch_opportunity.py) is put in code2wav's inbox ("q code2wav.in.put ... t=stream_chunk") and taken
("... .get"); the window's audio is put in code2wav's outbox ("q code2wav.out.put"), the k-th audio put for the k-th
window, and reaches the coordinator ("coord.recv ... from=code2wav n>0"). Splits each window's delay from the talker's
message to the coordinator into inbox (put to get), code2wav (get to the audio's put: waiting among the ready windows,
the replay, the copy to the host) and after (outbox to the coordinator), by percentile and by band of the whole delay.

usage: python3 c2w_window_path.py REPORT.sqlite --window window.txt
"""

from __future__ import annotations

import argparse
import collections
import re
import sqlite3
import statistics

from nsys_stage_ledger import session_window

MARK = 34
FIRST_WINDOW_CHUNK = 9


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
    for start, text, text_id in db.execute(
        "select start, text, textId from NVTX_EVENTS where eventType = ? order by start",
        (MARK,),
    ):
        label = text if text is not None else strings.get(text_id, "")
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
                rows.append((coord - put, get - put, out - get, coord - out))
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


if __name__ == "__main__":
    main()
