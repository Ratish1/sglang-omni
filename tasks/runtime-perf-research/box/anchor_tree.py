#!/usr/bin/env python3
"""Where one function's time goes on one thread, from a py-spy raw record (baseline.py pyspy pass).

usage: python3 anchor_tree.py PYSPY.txt THREAD_SUBSTRING ANCHOR_FUNCTION [--top 30] [--rate 100]

Samples of the thread whose stack holds the anchor: their count and ms, then frames below the anchor by inclusive
samples (a frame counts once per sample), the first frame below the anchor by samples, and leaf lines by samples.
"""

import argparse
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("thread")
    parser.add_argument("anchor")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--rate", type=float, default=100.0)
    args = parser.parse_args()
    thread_total, anchor_total = 0, 0
    inclusive, children, leaves = Counter(), Counter(), Counter()
    with open(args.record, encoding="utf-8") as source:
        for line in source:
            stack, _, count_text = line.rstrip("\n").rpartition(" ")
            frames = stack.split(";")
            if args.thread not in frames[0]:
                continue
            else:
                pass
            count = int(count_text)
            thread_total += count
            names = [frame.split(" (")[0] for frame in frames[1:]]
            if args.anchor not in names:
                continue
            else:
                pass
            anchor_total += count
            below = frames[1 + names.index(args.anchor) + 1 :]
            for frame in set(below):
                inclusive[frame] += count
            children[below[0] if below else "(self)"] += count
            leaves[below[-1] if below else "(self)"] += count
    ms = 1000.0 / args.rate
    print(
        f"thread samples {thread_total} ({thread_total * ms:.0f} ms), under {args.anchor} {anchor_total} ({anchor_total * ms:.0f} ms)"
    )
    for title, counter in (
        ("inclusive below the anchor", inclusive),
        ("first frame below", children),
        ("leaf", leaves),
    ):
        print(f"== {title}")
        for frame, count in counter.most_common(args.top):
            print(f"  {count:6d} {100 * count / max(anchor_total, 1):5.1f} %  {frame}")


if __name__ == "__main__":
    main()
