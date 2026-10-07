"""Where each inter-chunk gap at the coordinator comes from: the talker's cadence or what happens after the talker.

For each request, its window-completing codec messages (the talker sends the first 10 frames one message each, then
10 frames a message, so message 9 completes the first window and every later one the next) leave the talker on its
scheduler thread (probe mark "q talker_ar.out.put"), and its audio chunks reach the coordinator ("coord.recv ...
from=code2wav"). Chunk k's gap at the coordinator splits exactly into the talker's gap between its messages k-1 and
k, plus the change in the delay from a message leaving the talker to its audio arriving (code2wav's queue, decode,
copy and the hops). Reported by gap band, as census section L. For the talker's part, each gap also reports the
longest wait between the thinker's chunks reaching the talker's inbox for that request inside the talker's gap ("q
talker_ar.in.put ... t=stream_chunk"), and whether the thinker was still streaming to the request when the gap ended:
a talker that waits on the thinker's text shows long text waits while the thinker streams.

usage: python3 chunk_gap_origin.py REPORT.sqlite --window window.txt
"""

from __future__ import annotations

import argparse
import collections
import re
import sqlite3
import statistics

from nsys_stage_ledger import session_window

MARK = 34
FIRST_WINDOW_MESSAGE = 9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    sent: dict[str, list[int]] = collections.defaultdict(list)
    arrived: dict[str, list[int]] = collections.defaultdict(list)
    text: dict[str, list[int]] = collections.defaultdict(list)
    for start, text, text_id in db.execute(
        "select start, text, textId from NVTX_EVENTS where eventType = ? order by start", (MARK,)
    ):
        label = text if text is not None else strings.get(text_id, "")
        match = re.match(r"q talker_ar\.out\.put rid=(\S+) t=stream", label)
        if match:
            sent[match.group(1)].append(start)
            continue
        match = re.match(r"q talker_ar\.in\.put rid=(\S+) t=stream_chunk", label)
        if match:
            text[match.group(1)].append(start)
            continue
        match = re.match(r"coord\.recv rid=(\S+) from=code2wav n=(\d+)", label)
        if match and int(match.group(2)) > 0:
            arrived[match.group(1)].append(start)
        else:
            pass
    rows = []
    for request_id, arrivals in arrived.items():
        windows = sent.get(request_id, [])[FIRST_WINDOW_MESSAGE:]
        count = min(len(windows), len(arrivals))
        for k in range(1, count):
            if not t0 <= arrivals[k] <= t1:
                continue
            gap = arrivals[k] - arrivals[k - 1]
            talker = windows[k] - windows[k - 1]
            delay = (arrivals[k] - windows[k]) - (arrivals[k - 1] - windows[k - 1])
            inputs = text.get(request_id, [])
            inside = [windows[k - 1]] + [t for t in inputs if windows[k - 1] < t < windows[k]] + [windows[k]]
            longest_text_wait = max(b - a for a, b in zip(inside, inside[1:]))
            streaming = bool(inputs) and inputs[-1] > windows[k]
            rows.append(
                (gap, talker, delay, arrivals[k] - windows[k], longest_text_wait if streaming else 0, streaming)
            )
    rows.sort()
    if not rows:
        print("no gaps matched; check the probe marks")
        return
    else:
        pass
    n = len(rows)
    bands = (
        ("<= p50", 0, n // 2),
        ("p50 to p90", n // 2, int(n * 0.9)),
        ("p90 to p99", int(n * 0.9), int(n * 0.99)),
        ("> p99", int(n * 0.99), n),
    )
    print(f"window {(t1 - t0) / 1e9:.1f} s, {n} gaps (chunks after a request's first, last ones may be short)")
    print(
        f"  {'band':<12}{'gaps':>6}{'gap ms':>9}{'talker ms':>11}{'delay change':>14}{'delay ms':>10}"
        f"{'thinker streaming':>19}{'longest text wait ms':>22}"
    )
    for name, lo, hi in bands:
        band = rows[lo:hi]
        if not band:
            continue
        else:
            pass
        print(
            f"  {name:<12}{len(band):>6}{statistics.fmean(r[0] for r in band) / 1e6:>9.1f}"
            f"{statistics.fmean(r[1] for r in band) / 1e6:>11.1f}{statistics.fmean(r[2] for r in band) / 1e6:>14.1f}"
            f"{statistics.fmean(r[3] for r in band) / 1e6:>10.1f}"
            f"{100 * statistics.fmean(r[5] for r in band):>18.0f}%"
            f"{statistics.fmean(r[4] for r in band if r[5]) / 1e6 if any(r[5] for r in band) else 0:>22.1f}"
        )
    delays = sorted(r[3] for r in rows)
    print(
        "  talker message to audio at the coordinator (ms): "
        f"p50 {delays[n // 2] / 1e6:.1f}, p90 {delays[int(n * .9)] / 1e6:.1f}, p99 {delays[int(n * .99)] / 1e6:.1f}"
    )


if __name__ == "__main__":
    main()
