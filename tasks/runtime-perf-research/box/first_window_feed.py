#!/usr/bin/env python3
"""Whether a request's first window waits on the thinker's text or on the talker's steps, from one cell's events and
hop probe records (boots run with --probe and OMNI_HOP_PROBE_TALKER_BATCHES=1; queue records on).

usage: python3 first_window_feed.py <boot dir> [--cell c16_events] [--tail 0.05]

Per request, inside its window (talker first frame sent to code2wav's first window ready): the thinker's stream puts to
the talker for it (its text rows arriving), the talker decode steps that carry it (tb records), all talker decode
steps, and the longest gap between its text rows. A window whose own talker steps track its text puts is text-bound.
Printed for all requests (median) and the slowest --tail (mean), with how many requests joined.
"""

import argparse
import bisect
import collections
import glob
import json
import os
import statistics

FIRST_FRAME = ("talker_ar", "stage_first_stream_chunk_sent")
WINDOW_READY = ("code2wav", "code2wav_first_window_ready")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boot")
    parser.add_argument("--cell", default="c16_events")
    parser.add_argument("--tail", type=float, default=0.05)
    args = parser.parse_args()
    first = {}
    for path in glob.glob(os.path.join(args.boot, args.cell, "events", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                key = (record["stage"], record["event_name"])
                if key in (FIRST_FRAME, WINDOW_READY):
                    seen = first.setdefault(record["request_id"], {})
                    seen[key] = min(
                        seen.get(key, record["timestamp_ns"]), record["timestamp_ns"]
                    )
                else:
                    pass
    text_puts = collections.defaultdict(list)
    own_steps = collections.defaultdict(list)
    all_steps = []
    for path in glob.glob(os.path.join(args.boot, "probe", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if line.startswith("["):
                    item = json.loads(line)
                    if (
                        item[2] == "put"
                        and item[5] == "stream"
                        and item[6] == "talker_ar"
                        and item[4]
                    ):
                        text_puts[item[4]].append(item[0])
                    else:
                        pass
                elif '"k": "tb"' in line:
                    item = json.loads(line)
                    all_steps.append(item["ns"])
                    for rid in item["rids"]:
                        own_steps[rid].append(item["ns"])
                else:
                    pass
    for values in (*text_puts.values(), *own_steps.values(), all_steps):
        values.sort()

    def inside(values, start, end):
        return values[
            bisect.bisect_left(values, start) : bisect.bisect_right(values, end)
        ]

    rows = []
    for request_id, stamps in first.items():
        if FIRST_FRAME not in stamps or WINDOW_READY not in stamps:
            continue
        else:
            pass
        start, end = stamps[FIRST_FRAME], stamps[WINDOW_READY]
        puts = inside(text_puts.get(request_id, []), start, end)
        gaps = [b - a for a, b in zip([start] + puts, puts + [end])]
        rows.append(
            {
                "window": (end - start) / 1e6,
                "text_rows": len(puts),
                "own_steps": len(inside(own_steps.get(request_id, []), start, end)),
                "all_steps": len(inside(all_steps, start, end)),
                "longest_text_gap": max(gaps) / 1e6,
                "joined": request_id in text_puts and request_id in own_steps,
            }
        )
    if not rows:
        print(f"{args.boot} {args.cell}: no request with both stamps")
        return
    else:
        pass
    keys = ["window", "text_rows", "own_steps", "all_steps", "longest_text_gap"]
    slowest = sorted(rows, key=lambda row: -row["window"])[
        : max(1, int(len(rows) * args.tail))
    ]
    print(
        f"{args.boot} {args.cell}: {len(rows)} requests, {sum(row['joined'] for row in rows)} joined to text puts and talker rows"
    )
    print(
        "  all (median): "
        + "  ".join(
            f"{key} {statistics.median(row[key] for row in rows):.1f}" for key in keys
        )
    )
    print(
        f"  slowest {len(slowest)} (mean): "
        + "  ".join(
            f"{key} {statistics.fmean(row[key] for row in slowest):.1f}" for key in keys
        )
    )


if __name__ == "__main__":
    main()
