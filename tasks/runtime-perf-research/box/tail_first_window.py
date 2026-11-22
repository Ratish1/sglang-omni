#!/usr/bin/env python3
"""What fills a request's frames until code2wav's first window (talker first frame sent to code2wav first window
ready), from one cell's events and hop probe records (boots run with --probe, OMNI_HOP_PROBE_TALKER_BATCHES=1,
OMNI_HOP_PROBE_THINKER_STEPS=1 and, in a shared process, OMNI_HOP_PROBE_TURNS=1).

usage: python3 tail_first_window.py <boot dir> [--cell c16_events] [--tail 0.05]

Per request, inside its window: the talker's turn waits (tw, priority 1), its step results (tr: count and mean span),
the longest gap between consecutive talker step ends, and the thinker's extend and mixed steps (ts: count and summed
time). Printed for all requests (median) and for the slowest --tail by window (mean), then the slowest requests.
"""

import argparse
import bisect
import glob
import json
import os
import statistics

FIRST_FRAME = ("talker_ar", "stage_first_stream_chunk_sent")
WINDOW_READY = ("code2wav", "code2wav_first_window_ready")


def overlap(start, end, intervals, starts):
    total = 0
    index = max(0, bisect.bisect_left(starts, start) - 1)
    while index < len(intervals) and intervals[index][0] < end:
        total += max(0, min(end, intervals[index][1]) - max(start, intervals[index][0]))
        index += 1
    return total


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
    waits, talker_steps, thinker_steps = [], [], []
    for path in glob.glob(os.path.join(args.boot, "probe", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if '"k": "tw"' in line or '"k": "tr"' in line or '"k": "ts"' in line:
                    item = json.loads(line)
                    if item["k"] == "tw" and item["p"] == 1:
                        waits.append((item["ns0"], item["ns"]))
                    elif item["k"] == "tr":
                        talker_steps.append((item["ns0"], item["ns"]))
                    elif item["k"] == "ts" and item.get("mode") in ("EXTEND", "MIXED"):
                        thinker_steps.append((item["ns0"], item["ns"]))
                    else:
                        pass
                else:
                    pass
    for intervals in (waits, talker_steps, thinker_steps):
        intervals.sort()
    wait_starts = [interval[0] for interval in waits]
    thinker_starts = [interval[0] for interval in thinker_steps]
    talker_ends = sorted(interval[1] for interval in talker_steps)
    rows = []
    for request_id, stamps in first.items():
        if FIRST_FRAME not in stamps or WINDOW_READY not in stamps:
            continue
        else:
            pass
        start, end = stamps[FIRST_FRAME], stamps[WINDOW_READY]
        low, high = bisect.bisect_left(talker_ends, start), bisect.bisect_right(
            talker_ends, end
        )
        ends = talker_ends[low:high]
        gaps = [b - a for a, b in zip([start] + ends, ends + [end])]
        inside = [s for s in talker_steps if start <= s[1] <= end]
        rows.append(
            {
                "window": (end - start) / 1e6,
                "turn_wait": overlap(start, end, waits, wait_starts) / 1e6,
                "talker_steps": len(inside),
                "talker_step_ms": (
                    statistics.fmean((b - a) / 1e6 for a, b in inside)
                    if inside
                    else 0.0
                ),
                "longest_gap": max(gaps) / 1e6,
                "thinker_prefill_steps": sum(
                    1 for s in thinker_steps if s[0] < end and s[1] > start
                ),
                "thinker_prefill_ms": overlap(start, end, thinker_steps, thinker_starts)
                / 1e6,
                "request": request_id,
            }
        )
    if not rows:
        print(f"{args.boot} {args.cell}: no request with both stamps")
        return
    else:
        pass
    keys = [
        "window",
        "turn_wait",
        "talker_steps",
        "talker_step_ms",
        "longest_gap",
        "thinker_prefill_steps",
        "thinker_prefill_ms",
    ]
    slowest = sorted(rows, key=lambda row: -row["window"])[
        : max(1, int(len(rows) * args.tail))
    ]
    print(
        f"{args.boot} {args.cell}: {len(rows)} requests, turn waits {len(waits)}, talker steps {len(talker_steps)}"
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
    for row in slowest[:8]:
        print("    " + "  ".join(f"{key} {row[key]:.1f}" for key in keys))


if __name__ == "__main__":
    main()
