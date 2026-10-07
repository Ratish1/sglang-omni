"""How many requests' code2wav windows wait in its inbox whenever it starts a window, from a probed census.

The talker sends a request's first 10 codec frames one message each, then one message per 10 frames, so a request's
stream chunk number 9 completes its first window (10 frames) and every later chunk completes the next one (20, 30,
then 35 frames with the context saturated). At each threshold decode (c2w.decode final=0) on the code2wav thread,
counts the other requests whose window-completing chunk was put in code2wav's inbox and not yet taken: the windows
one replay could have decoded together had the thread drained its inbox first, all frames and same frames. Also the
time each window's chunk waited in the inbox.

usage: python3 c2w_batch_opportunity.py REPORT.sqlite --window window.txt
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3

from nsys_stage_ledger import session_window

PUSH_POP = 59
MARK = 34
FIRST_WINDOW_CHUNK = 9


def window_frames(chunk_index: int) -> int:
    return {9: 10, 10: 20, 11: 30}.get(chunk_index, 35)


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
    decodes: list[tuple[int, int, str]] = []
    replay_rows: collections.Counter = collections.Counter()
    for start, end, text, text_id, kind in db.execute(
        "select start, end, text, textId, eventType from NVTX_EVENTS where eventType in (?, ?) order by start",
        (PUSH_POP, MARK),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if kind == MARK:
            match = re.match(r"q code2wav\.in\.(put|get) rid=(\S+) t=stream_chunk", label)
            if match:
                (puts if match.group(1) == "put" else gets)[match.group(2)].append(start)
            else:
                pass
        else:
            match = re.match(r"c2w\.decode rid=(\S+) final=0(?: rows=(\d+))?", label)
            if match and t0 <= start <= t1:
                decodes.append((start, end, match.group(1)))
                replay_rows[int(match.group(2) or 1)] += 1
            else:
                pass
    # every window-completing chunk: (put time, get time, request, frames)
    windows = []
    for request_id, request_puts in puts.items():
        request_gets = gets.get(request_id, [])
        for index, put_time in enumerate(request_puts):
            if index >= FIRST_WINDOW_CHUNK and index < len(request_gets):
                windows.append((put_time, request_gets[index], request_id, window_frames(index)))
            else:
                pass
    windows.sort()
    put_times = [w[0] for w in windows]
    rows_all = collections.Counter()
    rows_same = collections.Counter()
    waits = []
    for start, _, request_id in decodes:
        upto = bisect.bisect_right(put_times, start)
        queued = [w for w in windows[:upto] if w[1] > start and w[2] != request_id]
        mine = [w for w in windows[:upto] if w[2] == request_id and w[1] <= start]
        frames = mine[-1][3] if mine else 35
        rows_all[1 + len(queued)] += 1
        rows_same[1 + sum(1 for w in queued if w[3] == frames)] += 1
        if mine:
            waits.append((mine[-1][1] - mine[-1][0]) / 1e6)
        else:
            pass
    total = sum(rows_all.values())
    print(f"window {(t1 - t0) / 1e9:.1f} s, threshold decodes {total}, window chunks {len(windows)}")
    windows_decoded = sum(rows * count for rows, count in replay_rows.items())
    print(
        f"  replays {sum(replay_rows.values())} decoding {windows_decoded} windows; rows per replay: "
        + ", ".join(f"{rows}: {count}" for rows, count in sorted(replay_rows.items()))
    )
    for name, counter in (("any frames", rows_all), ("same frames", rows_same)):
        mean = sum(k * v for k, v in counter.items()) / max(total, 1)
        shares = ", ".join(f"{k}: {100 * v / total:.1f}%" for k, v in sorted(counter.items()))
        print(f"  rows one replay could take ({name}): mean {mean:.2f}; {shares}")
        replays = sum(v / k for k, v in counter.items())
        print(f"    replays if every decode took its queued windows: {replays:.0f} of {total}")
    waits.sort()
    if waits:
        print(
            "  inbox wait of the decoded window's chunk (ms): "
            f"p50 {waits[len(waits) // 2]:.2f}, p90 {waits[int(len(waits) * .9)]:.2f}, "
            f"p99 {waits[int(len(waits) * .99)]:.2f}, max {waits[-1]:.2f}"
        )
    else:
        pass


if __name__ == "__main__":
    main()
