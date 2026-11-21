#!/usr/bin/env python3
"""How long each engine of a shared process waits for its turn, and what the leaver was doing, from hop probe "tw" and
"th" records (boots run with --probe and OMNI_HOP_PROBE_TURNS=1).

usage: python3 turn_waits.py <boot dir> [--cell c64_events] [--top 15]

Per waiter priority (1 the talker, 0 the thinker): waits a second, waited ms a second, and the wait p50, p90, p99 and
max. Each talker wait is joined to the leave that handed it the turn (the last "th" ending at or before the wait's end,
within 2 ms): by the leaver's site, the waits, their summed ms and the leaver's mean hold span.
"""

import argparse
import bisect
import collections
import glob
import json
import os


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boot")
    parser.add_argument("--cell", default="c64_events")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    with open(
        os.path.join(args.boot, args.cell, "window.json"), encoding="utf-8"
    ) as source:
        window = json.load(source)
    start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
    seconds = window["end"] - window["start"]
    waits, handoffs = [], []
    for path in glob.glob(os.path.join(args.boot, "probe", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if '"k": "tw"' in line or '"k": "th"' in line:
                    item = json.loads(line)
                    if start_ns <= item["ns"] <= end_ns:
                        (waits if item["k"] == "tw" else handoffs).append(item)
                    else:
                        pass
                else:
                    pass
    print(
        f"{args.boot} {args.cell}: {seconds:.1f} s, {len(waits)} waits, {len(handoffs)} handing leaves"
    )
    for priority in (1, 0):
        spans = [
            (item["ns"] - item["ns0"]) / 1e6 for item in waits if item["p"] == priority
        ]
        print(
            f"  waiter priority {priority}: {len(spans) / seconds:7.1f} waits/s, {sum(spans) / seconds:6.1f} ms/s waited, "
            f"p50 {quantile(spans, 0.5):.2f} p90 {quantile(spans, 0.9):.2f} p99 {quantile(spans, 0.99):.2f} "
            f"max {max(spans, default=0.0):.2f} ms"
        )
    handoffs.sort(key=lambda item: item["ns"])
    ends = [item["ns"] for item in handoffs]
    by_site = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for item in waits:
        if item["p"] != 1:
            continue
        else:
            pass
        index = bisect.bisect_right(ends, item["ns"]) - 1
        if (
            index >= 0
            and item["ns"] - ends[index] <= 2_000_000
            and handoffs[index]["p"] == 0
        ):
            site = handoffs[index]["site"]
            hold = (handoffs[index]["ns"] - handoffs[index]["ns0"]) / 1e6
        else:
            site, hold = "(no thinker leave joined)", 0.0
        entry = by_site[site]
        entry[0] += 1
        entry[1] += (item["ns"] - item["ns0"]) / 1e6
        entry[2] += hold
    print(
        "  talker waits by the thinker leave that ended them: waits, ms/s waited, mean thinker hold ms, site"
    )
    for site, (count, waited, held) in sorted(
        by_site.items(), key=lambda pair: -pair[1][1]
    )[: args.top]:
        print(f"    {count:6d} {waited / seconds:7.2f} {held / count:8.2f}  {site}")


if __name__ == "__main__":
    main()
