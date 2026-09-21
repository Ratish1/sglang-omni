"""V-e12: replay a vocoder census against a measured cost grid, today and with merging.

Inputs: a request_flow probe prefix (kind "cohort" and "replay" records) and the grid JSON
of vocoder_valid_width_bench.py (captured graph ms per "<width>x<rows>" key, today's
decode and the valid width variant).

today    every recorded replay at its (width, bucket) key, today's cost.
merged   bootstraps of 1 or 2 rows: one replay at the smallest window width that holds
         them, instead of the window chain. Follow ups: cohorts a worker ran back to back
         (the next one started within --gap-ms of the previous one's end, so they came
         out of one collection) become one replay at their largest width, rows summed,
         capped at 8 rows. Costs from the variant's column.

usage: python vocoder_merge_sim.py <flow prefix> <grid.json> [--gap-ms 0.5]
"""

import argparse
import glob
import json
from collections import defaultdict

BUCKETS = (1, 2, 4, 8)
WINDOWS = (1, 2, 4, 8, 16, 32, 64)


def bucket(rows):
    return next(size for size in BUCKETS if size >= rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    parser.add_argument("grid")
    parser.add_argument("--gap-ms", type=float, default=0.5)
    args = parser.parse_args()
    with open(args.grid) as handle:
        grid = json.load(handle)
    cohorts, replays = [], []
    for path in sorted(glob.glob(f"{args.prefix}.*.jsonl")):
        with open(path) as lines:
            for line in lines:
                record = json.loads(line)
                if record["kind"] == "cohort":
                    cohorts.append(record)
                elif record["kind"] == "replay" and record["hit"]:
                    replays.append(record)

    def cost(width, rows, column):
        return grid[f"{width}x{bucket(rows)}"][column]

    today_ms = sum(cost(r["width"], r["rows"], "today_ms") for r in replays)
    span_s = max(c["t"] for c in cohorts) - min(c["t"] for c in cohorts)

    merged_ms, merged_replays = 0.0, 0
    boot_today = boot_merged = 0.0
    boots = [c for c in cohorts if c["initial_stream"]]
    for c in boots:
        chain = sum(cost(w, c["rows"], "today_ms") for w in c["split"] or [c["width"]])
        boot_today += chain
        if c["rows"] <= 2 and c["width"] <= WINDOWS[-1]:
            window = next(w for w in WINDOWS if w >= c["width"])
            one = cost(window, c["rows"], "valid_ms")
            # note(ratish): a chain that is already one window keeps today's cost
            one = min(one, chain) if len(c["split"] or [0]) == 1 else one
            boot_merged += one
            merged_replays += 1
        else:
            boot_merged += chain
            merged_replays += len(c["split"] or [0])
    merged_ms += boot_merged

    by_thread = defaultdict(list)
    for c in cohorts:
        if not c["initial_stream"]:
            by_thread[c["thread"]].append(c)
    follow_today = follow_merged = 0.0
    group_rows = []
    for items in by_thread.values():
        items.sort(key=lambda c: c["t"] - c["wall_s"])
        group = []
        for c in items:
            follow_today += cost(c["width"], c["rows"], "today_ms")
            start = c["t"] - c["wall_s"]
            fits = (
                group
                and (start - group[-1]["t"]) * 1000.0 <= args.gap_ms
                and sum(g["rows"] for g in group) + c["rows"] <= BUCKETS[-1]
            )
            if not fits and group:
                rows = sum(g["rows"] for g in group)
                follow_merged += cost(max(g["width"] for g in group), rows, "valid_ms")
                group_rows.append(rows)
                group = []
            group.append(c)
        if group:
            rows = sum(g["rows"] for g in group)
            follow_merged += cost(max(g["width"] for g in group), rows, "valid_ms")
            group_rows.append(rows)
    merged_ms += follow_merged
    merged_replays += len(group_rows)

    print(f"census span {span_s:.1f} s, {len(cohorts)} cohorts, {len(replays)} replays")
    print(f"today   all replays           {today_ms / 1000:8.2f} s device")
    print(
        f"  bootstraps {len(boots):5d}: chain {boot_today / 1000:6.2f} s -> one replay "
        f"{boot_merged / 1000:6.2f} s ({100 * (boot_merged / boot_today - 1):+.1f}%), "
        f"mean per bootstrap {boot_today / len(boots):.2f} -> {boot_merged / len(boots):.2f} ms"
    )
    follow = sum(len(v) for v in by_thread.values())
    print(
        f"  follow ups {follow:5d}: {follow_today / 1000:6.2f} s -> {len(group_rows)} replays "
        f"{follow_merged / 1000:6.2f} s ({100 * (follow_merged / follow_today - 1):+.1f}%), "
        f"rows per replay {sum(group_rows) / len(group_rows):.2f}"
    )
    print(
        f"merged  all replays           {merged_ms / 1000:8.2f} s device "
        f"({100 * (merged_ms / today_ms - 1):+.1f}%), replays {len(replays)} -> "
        f"{merged_replays}, {100 * (today_ms - merged_ms) / 1000 / span_s:.1f}% of the span"
    )


if __name__ == "__main__":
    main()
