"""How much of each Fun-CosyVoice3 Flow graph replay is padding, from one nsys report taken with cosy_profile_probe and
the server log of the same run.

The prefix and whole history graphs are captured per frame tier (the tier lists the server logs at capture) and a
replay runs the smallest tier holding its real frames (the frames= field of the probe's flow.prefix_graph and
flow.whole_history_graph ranges). Per graph kind: replays, real frames, tier frames, the padded share of the tier
frames, the replays by tier, and each replay's own device time per tier (the graph replay the vocoder thread
launched inside the range).

usage: python3 graph_padding_c20261020.py REPORT.sqlite SERVE.log [--skip-head-s S] [--skip-tail-s S]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics

from cosy_ledger import RANGE_EVENTS, has_table, pid_of

TIERS = re.compile(r"Fun-CosyVoice3 (prefix|whole history) graphs captured: tiers=\[([0-9, ]+)\]")
RANGE = re.compile(r"flow\.(prefix_graph|whole_history_graph) rows=(\d+) frames=(\d+)")
KIND = {"prefix_graph": "prefix", "whole_history_graph": "whole history"}
MS = 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("serve_log")
    parser.add_argument("--skip-head-s", type=float, default=1.0)
    parser.add_argument("--skip-tail-s", type=float, default=1.0)
    args = parser.parse_args()
    tiers: dict[str, list[int]] = {}
    with open(args.serve_log) as log:
        for line in log:
            match = TIERS.search(line)
            if match:
                tiers[match[1]] = sorted(int(t) for t in match[2].split(","))
            else:
                pass
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    graph_work = []
    if has_table(db, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        graph_work = list(
            db.execute("select start, end, correlationId, globalPid from CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        )
    else:
        pass
    t0 = min(w[0] for w in graph_work) + int(args.skip_head_s * 1e9)
    t1 = max(w[1] for w in graph_work) - int(args.skip_tail_s * 1e9)
    launches: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for start, tid, corr in db.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        launches[tid].append((start, corr))
    for tid in launches:
        launches[tid].sort()
    graph_time = {(pid_of(g), c): e - s for s, e, c, g in graph_work}

    replays: dict[str, list[tuple[int, int, float]]] = collections.defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
        f"select start, end, globalTid, text, textId from NVTX_EVENTS where eventType in {RANGE_EVENTS} and end is not null"
    ):
        label = text if text is not None else strings.get(text_id, "")
        match = RANGE.match(label or "")
        if not match or not t0 <= start <= t1:
            continue
        else:
            pass
        kind = KIND[match[1]]
        frames = int(match[3])
        kind_tiers = tiers.get(kind, [])
        index = bisect.bisect_left(kind_tiers, frames)
        if index == len(kind_tiers):
            continue
        else:
            pass
        calls = launches[tid]
        a = bisect.bisect_left(calls, (start, -1))
        b = bisect.bisect_right(calls, (end, 1 << 62))
        device = sum(graph_time.get((pid_of(tid), corr), 0) for _, corr in calls[a:b]) / MS
        replays[kind].append((frames, kind_tiers[index], device))

    for kind, rows in sorted(replays.items()):
        real = sum(r[0] for r in rows)
        padded = sum(r[1] for r in rows)
        device = sum(r[2] for r in rows)
        print(f"\n{kind} graph: tiers {tiers.get(kind)}")
        print(
            f"  {len(rows)} replays, real frames {real}, tier frames {padded}, padded share "
            f"{100 * (padded - real) / padded:.1f} %, device {device:.1f} ms"
        )
        by_tier: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)
        for frames, tier, ms in rows:
            by_tier[tier].append((frames, ms))
        print("  tier: replays, mean real frames, padded share, device ms per replay, device ms summed")
        for tier, items in sorted(by_tier.items()):
            mean_frames = statistics.mean(f for f, _ in items)
            mean_ms = statistics.mean(m for _, m in items)
            print(
                f"  {tier:6d}: {len(items):5d} {mean_frames:8.1f} {100 * (1 - mean_frames / tier):6.1f} % "
                f"{mean_ms:7.2f} {sum(m for _, m in items):9.1f}"
            )


if __name__ == "__main__":
    main()
