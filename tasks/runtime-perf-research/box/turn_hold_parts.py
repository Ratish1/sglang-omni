#!/usr/bin/env python3
"""What fills the thinker's turn holds that a waiting talker sits through, from hop probe "th" records (the thinker's
leaves that hand the turn on) and the thinker step records "tq", "tg", "ts" and "tp" (boots run with --probe,
OMNI_HOP_PROBE_TURNS=1 and OMNI_HOP_PROBE_THINKER_STEPS=1).

usage: python3 turn_hold_parts.py <boot dir> [--cell c64_events] [--long-ms 10] [--top 12]

Each hold's time is split by overlap with the thinker's input processing (tq, request builds inline), batch selection
(tg, recorded for extend and mixed batches and any over 2 ms), the step (ts, build, launch and finalize until its
release) and result processing (tp); the rest is loop work between them. Printed: the hold distribution, mean ms per
part over all holds and over holds past --long-ms, and the longest holds with the step's mode and extend tokens.
"""

import argparse
import glob
import json
import os
import statistics

PARTS = (("tq", "inputs"), ("tg", "select"), ("ts", "step"), ("tp", "result"))


def overlap_ms(start, end, intervals):
    total = 0
    for interval_start, interval_end, _ in intervals:
        if interval_end <= start:
            continue
        elif interval_start >= end:
            break
        else:
            total += min(end, interval_end) - max(start, interval_start)
    return total / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boot")
    parser.add_argument("--cell", default="c64_events")
    parser.add_argument("--long-ms", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    with open(
        os.path.join(args.boot, args.cell, "window.json"), encoding="utf-8"
    ) as source:
        window = json.load(source)
    start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
    holds, phases = [], {kind: [] for kind, _ in PARTS}
    for path in glob.glob(os.path.join(args.boot, "probe", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if '"k": "th"' in line:
                    item = json.loads(line)
                    if item["p"] == 0 and start_ns <= item["ns"] <= end_ns:
                        holds.append(item)
                    else:
                        pass
                elif any(f'"k": "{kind}"' in line for kind in phases):
                    item = json.loads(line)
                    if start_ns - 1e9 <= item["ns"] <= end_ns + 1e9:
                        phases[item["k"]].append((item["ns0"], item["ns"], item))
                    else:
                        pass
                else:
                    pass
    for intervals in phases.values():
        intervals.sort(key=lambda interval: interval[0])
    rows = []
    for hold in holds:
        length = (hold["ns"] - hold["ns0"]) / 1e6
        split = {
            name: overlap_ms(hold["ns0"], hold["ns"], phases[kind])
            for kind, name in PARTS
        }
        split["other"] = max(0.0, length - sum(split.values()))
        steps = [
            item
            for interval_start, interval_end, item in phases["ts"]
            if interval_start < hold["ns"] and interval_end > hold["ns0"]
        ]
        shape = (
            f"{steps[-1].get('mode', '?')} ext {steps[-1].get('ext', '?')}"
            if steps
            else "no step"
        )
        rows.append((length, split, shape, hold["site"]))
    lengths = sorted(row[0] for row in rows)
    if not lengths:
        print(f"{args.boot} {args.cell}: no thinker holds")
        return
    else:
        pass
    pick = lambda q: lengths[min(len(lengths) - 1, int(q * len(lengths)))]
    print(
        f"{args.boot} {args.cell}: {len(rows)} thinker holds handed to a waiter, ms p50 {pick(0.5):.2f} "
        f"p90 {pick(0.9):.2f} p99 {pick(0.99):.2f} max {lengths[-1]:.2f}"
    )
    names = [name for _, name in PARTS] + ["other"]
    for title, chosen in (
        ("all holds", rows),
        (
            f"holds over {args.long_ms:.0f} ms",
            [row for row in rows if row[0] > args.long_ms],
        ),
    ):
        if not chosen:
            continue
        else:
            pass
        means = "  ".join(
            f"{name} {statistics.fmean(row[1][name] for row in chosen):6.2f}"
            for name in names
        )
        print(
            f"  {title} ({len(chosen)}): mean {statistics.fmean(row[0] for row in chosen):6.2f} ms = {means}"
        )
    print("  longest holds: ms, parts, step, site")
    for length, split, shape, site in sorted(rows, key=lambda row: -row[0])[: args.top]:
        parts = " ".join(f"{name} {split[name]:.1f}" for name in names)
        print(f"    {length:7.2f}  {parts}  [{shape}]  {site}")


if __name__ == "__main__":
    main()
