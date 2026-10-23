"""Every NVTX range kind on one engine scheduler thread inside the measured window.

A kind is the label's leading words up to the first key=value word, so "sched.batch decode bs=32"
is "sched.batch decode". Nested ranges overlap their parents, so the window shares do not sum to
100 %. GIL ranges are left out.

usage: python3 talker_range_kinds.py REPORT.sqlite --window window.txt [--thread scheduler-talker_ar]
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import statistics

from nsys_stage_ledger import session_window

PUSH_POP = 59
MARK = 34


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    parser.add_argument("--thread", default="scheduler-talker_ar")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    threads = set()
    for thread, text, text_id in db.execute(
        "select globalTid, text, textId from NVTX_EVENTS where eventType = ?", (MARK,)
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith(f"thread name={args.thread} "):
            threads.add(thread)
        else:
            pass
    kinds: dict[str, list[float]] = collections.defaultdict(list)
    for thread in threads:
        for start, end, text, text_id in db.execute(
            "select start, end, text, textId from NVTX_EVENTS where globalTid = ? and eventType = ? "
            "and end is not null and start between ? and ?",
            (thread, PUSH_POP, t0, t1),
        ):
            label = text if text is not None else strings.get(text_id, "")
            if label.startswith(("Holding GIL", "Waiting for GIL")):
                continue
            else:
                pass
            words = []
            for word in label.split(" "):
                if "=" in word:
                    break
                else:
                    words.append(word)
            kinds[" ".join(words[:2])].append((end - start) / 1e6)
    span = (t1 - t0) / 1e6
    print(f"window {span / 1e3:.1f} s, thread {args.thread}")
    print(
        f"  {'kind':28s}{'n':>7s}{'mean':>8s}{'p50':>8s}{'p90':>8s}{'total s':>9s}{'% window':>9s}"
    )
    for kind, walls in sorted(kinds.items(), key=lambda item: -sum(item[1])):
        walls.sort()
        print(
            f"  {kind:28s}{len(walls):7d}{statistics.fmean(walls):8.2f}"
            f"{walls[len(walls) // 2]:8.2f}{walls[int(len(walls) * 0.9)]:8.2f}"
            f"{sum(walls) / 1e3:9.2f}{100 * sum(walls) / span:9.1f}"
        )


if __name__ == "__main__":
    main()
