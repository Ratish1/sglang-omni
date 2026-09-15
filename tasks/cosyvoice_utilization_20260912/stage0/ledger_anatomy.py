#!/usr/bin/env python3
"""Where a streaming or buffered vocoder step spends its frames and its time,
from a call ledger directory written by call_ledger/cosy_call_ledger.py.

  python tasks/cosyvoice_utilization_20260912/stage0/ledger_anatomy.py <ledger dir>

Standard library only.

1. Hop frames: the frames every hop recomputes (2 (P + O + H) per row) against
   the frames a hop over cached finished chunks would run (2 (P + H) on the
   first hop, 2 H after it), summed over the run.
2. Step time: per streaming step, the host time of its Flow call, of its HiFT
   calls and of the rest, summed over the run.
3. Padding: hop and final calls split by rows x widest / total frames, with
   their host time per call and per thousand total frames.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys


def pct(values, p):
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    return ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1]


def fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    directory = sys.argv[1]
    entries = []
    for path in glob.glob(os.path.join(directory, "ledger_*.jsonl")):
        with open(path) as handle:
            entries.extend(json.loads(line) for line in handle if line.strip())
    steps = {e["id"]: e for e in entries if e["kind"] == "step"}
    children = collections.defaultdict(list)
    for entry in entries:
        if entry["kind"] != "step" and entry.get("step") is not None:
            children[entry["step"]].append(entry)

    recomputed = cached = first_hop_rows = later_rows = 0
    for step in steps.values():
        if step["plan"] != "causal_window":
            continue
        for participant in step["participants"]:
            prompt = participant["prompt_tokens"] or 0
            offset, hop = participant["token_offset"], participant["hop_len"]
            recomputed += 2 * (prompt + offset + hop)
            if offset == 0:
                cached += 2 * (prompt + hop)
                first_hop_rows += 1
            else:
                cached += 2 * hop
                later_rows += 1
    final_frames = sum(
        e["total_frames"]
        for e in entries
        if e["kind"] == "final" and e.get("step") is not None
    )
    if recomputed:
        print("1. hop frames over the run")
        print(f"   hop rows: first hops {first_hop_rows}, later hops {later_rows}")
        print(f"   frames every hop runs today: {recomputed}")
        print(
            f"   frames over cached finished chunks: {cached} ({100.0 * cached / recomputed:.1f} percent)"
        )
        print(f"   final call frames (bidirectional, unchanged): {final_frames}")
        print(
            f"   hop plus final frames today {recomputed + final_frames}, "
            f"with the cache {cached + final_frames} "
            f"({100.0 * (cached + final_frames) / (recomputed + final_frames):.1f} percent)"
        )

    split = collections.defaultdict(float)
    for step_id, step in steps.items():
        flow = sum(
            c["host_ms"] for c in children[step_id] if c["kind"] in ("hop", "final")
        )
        hift = sum(c["host_ms"] for c in children[step_id] if c["kind"] == "hift")
        split[f"{step['plan']} flow"] += flow
        split[f"{step['plan']} hift"] += hift
        split[f"{step['plan']} rest"] += step["host_ms"] - flow - hift
        split["all steps"] += step["host_ms"]
    buffered_flow = sum(e["host_ms"] for e in entries if e["kind"] == "buffered_flow")
    buffered_hift = sum(e["host_ms"] for e in entries if e["kind"] == "hift_batch")
    print("\n2. host time over the run, seconds")
    for key, value in split.items():
        print(f"   {key:<24} {value / 1e3:8.1f}")
    if buffered_flow:
        print(f"   buffered flow calls      {buffered_flow / 1e3:8.1f}")
        print(f"   buffered hift batches    {buffered_hift / 1e3:8.1f}")

    print("\n3. calls by rows x widest / total frames")
    print(
        f"   {'kind':<14}{'pad ratio':<12}{'calls':>6}{'rows p50':>9}{'widest p50':>11}{'host ms p50':>12}{'p95':>8}{'ms per 1k frames p50':>22}"
    )
    for kind in ("hop", "final", "buffered_flow"):
        calls = [
            e
            for e in entries
            if e["kind"] == kind
            and (kind == "buffered_flow" or e.get("step") is not None)
        ]
        for label, low, high in (
            ("< 1.5", 0, 1.5),
            ("1.5 to 3", 1.5, 3),
            (">= 3", 3, math.inf),
        ):
            bucket = [
                c
                for c in calls
                if low <= c["rows"] * c["widest_frames"] / c["total_frames"] < high
            ]
            if not bucket:
                continue
            print(
                f"   {kind:<14}{label:<12}{len(bucket):>6}"
                f"{fmt(pct([c['rows'] for c in bucket], 50), 0):>9}"
                f"{fmt(pct([c['widest_frames'] for c in bucket], 50), 0):>11}"
                f"{fmt(pct([c['host_ms'] for c in bucket], 50)):>12}"
                f"{fmt(pct([c['host_ms'] for c in bucket], 95)):>8}"
                f"{fmt(pct([1e3 * c['host_ms'] / c['total_frames'] for c in bucket], 50)):>22}"
            )

    wide = [
        e
        for e in entries
        if e["kind"] == "hop"
        and e.get("step") is not None
        and e["widest_frames"] >= 2000
    ]
    if wide:
        print(
            f"\n   hop calls with a row of 2,000 frames or more: {len(wide)}, host ms p50 {fmt(pct([c['host_ms'] for c in wide], 50))}"
        )


if __name__ == "__main__":
    main()
