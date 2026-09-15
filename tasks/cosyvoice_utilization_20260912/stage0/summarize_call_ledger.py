#!/usr/bin/env python3
"""Tables from a call ledger directory written by call_ledger/cosy_call_ledger.py.

  python tasks/cosyvoice_utilization_20260912/stage0/summarize_call_ledger.py \
      <ledger dir> [--json out.json]

Standard library only. Hop and final calls outside a scheduler step (the boot
warmup) are listed separately and left out of the call tables.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os


def pct(values, p):
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
    return ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1]


def fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def spread(values, digits=1):
    return (
        "/".join(fmt(pct(values, p), digits) for p in (50, 95))
        + f"/{fmt(max(values) if values else None, digits)}"
    )


def table(title, header, rows):
    print(f"\n{title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(str(cell) for cell in row) + " |")


def rows_bucket(rows):
    for low, high in ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16)):
        if low <= rows <= high:
            return f"{low}" if low == high else f"{low}-{high}"
    return ">16"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger_dir")
    parser.add_argument("--json")
    args = parser.parse_args()

    entries = []
    for path in glob.glob(os.path.join(args.ledger_dir, "ledger_*.jsonl")):
        with open(path) as handle:
            entries.extend(json.loads(line) for line in handle if line.strip())
    by_kind = collections.defaultdict(list)
    warmup = collections.Counter()
    for entry in entries:
        if entry["kind"] in ("hop", "final") and entry.get("step") is None:
            warmup[entry["kind"]] += 1
            continue
        by_kind[entry["kind"]].append(entry)
    print(f"entries {len(entries)}; outside a step (warmup): {dict(warmup)}")

    flow_rows = []
    for kind in ("hop", "final", "buffered_flow"):
        calls = by_kind.get(kind, [])
        if not calls:
            continue
        pad = [
            c["rows"] * c["widest_frames"] / c["total_frames"]
            for c in calls
            if c.get("total_frames")
        ]
        ratio = [
            c["gpu_ms"] / c["host_ms"]
            for c in calls
            if c.get("gpu_ms") and c.get("host_ms")
        ]
        flow_rows.append(
            [
                kind,
                len(calls),
                spread([c["rows"] for c in calls], 0),
                spread([c["total_frames"] for c in calls], 0),
                spread([c["widest_frames"] for c in calls], 0),
                spread(pad, 2),
                spread([c["host_ms"] for c in calls]),
                spread([c.get("gpu_ms") for c in calls if c.get("gpu_ms") is not None]),
                fmt(pct(ratio, 50), 2),
            ]
        )
    table(
        "Flow calls (p50/p95/max)",
        [
            "kind",
            "calls",
            "rows",
            "total frames",
            "widest frames",
            "rows x widest / total",
            "host ms",
            "gpu ms",
            "gpu/host p50",
        ],
        flow_rows,
    )

    histogram = []
    for kind in ("hop", "final", "buffered_flow"):
        counts = collections.Counter(
            rows_bucket(c["rows"]) for c in by_kind.get(kind, [])
        )
        histogram.append(
            [kind] + [counts.get(b, 0) for b in ("1", "2", "3-4", "5-8", "9-16", ">16")]
        )
    table(
        "Rows per Flow call", ["kind", "1", "2", "3-4", "5-8", "9-16", ">16"], histogram
    )

    distinct = []
    for kind in ("hop", "final", "buffered_flow"):
        calls = by_kind.get(kind, [])
        if not calls:
            continue
        distinct.append(
            [
                kind,
                len({(c["rows"], c["widest_frames"]) for c in calls}),
                len({(c["rows"], -(-c["widest_frames"] // 16) * 16) for c in calls}),
                len({c["total_frames"] for c in calls}),
                len({(c["rows"], tuple(sorted(c["row_frames"]))) for c in calls}),
            ]
        )
    table(
        "Distinct shapes the calls took",
        [
            "kind",
            "(rows, widest)",
            "(rows, widest to 16)",
            "total frames",
            "(rows, row frames multiset)",
        ],
        distinct,
    )

    steps = by_kind.get("step", [])
    if steps:
        participants = [p for s in steps for p in s["participants"]]
        plans = collections.Counter(s["plan"] for s in steps)
        started = [
            p["wait_ms"]
            for p in participants
            if p["started"] and p["wait_ms"] is not None
        ]
        unstarted = [
            p["wait_ms"]
            for p in participants
            if not p["started"] and p["wait_ms"] is not None
        ]
        table(
            "Scheduler steps",
            ["read", "value"],
            [
                ["steps by plan", dict(plans)],
                [
                    "participants per step p50/p95/max",
                    spread([len(s["participants"]) for s in steps], 0),
                ],
                [
                    "hop_len counts",
                    dict(collections.Counter(p["hop_len"] for p in participants)),
                ],
                [
                    "token_offset p50/p95/max",
                    spread([p["token_offset"] for p in participants], 0),
                ],
                [
                    "prompt tokens p50/p95/max",
                    spread(
                        [
                            p["prompt_tokens"]
                            for p in participants
                            if p["prompt_tokens"] is not None
                        ],
                        0,
                    ),
                ],
                ["wait since ready ms, started, p50/p95/max", spread(started)],
                ["wait since ready ms, first hop, p50/p95/max", spread(unstarted)],
                ["step host ms p50/p95/max", spread([s["host_ms"] for s in steps])],
                [
                    "step gpu ms p50/p95/max",
                    spread([s["gpu_ms"] for s in steps if s.get("gpu_ms") is not None]),
                ],
            ],
        )
        hop_keys = collections.Counter()
        for s in steps:
            if "causal" in s["plan"]:
                hop_keys[
                    (
                        len(s["participants"]),
                        max(p["hop_len"] for p in s["participants"]),
                    )
                ] += 1
        print(f"\ndistinct (rows, largest hop_len) over hop steps: {len(hop_keys)}")

    runs = by_kind.get("graph_run", [])
    if runs:
        hits = [r for r in runs if r.get("hit")]
        absent = [r for r in runs if not r.get("hit") and not r["captured"]]
        mismatch = [r for r in runs if not r.get("hit") and r["captured"]]
        missed = collections.Counter((r["batch"], r["bucket"]) for r in absent)
        table(
            "Buffered graph runner",
            ["read", "value"],
            [
                ["calls", len(runs)],
                ["hits", len(hits)],
                ["misses, key not captured", len(absent)],
                ["misses, key captured but inputs differ", len(mismatch)],
                [
                    "padded frames on hits",
                    sum(r["batch"] * (r["bucket"] - r["frames"]) for r in hits),
                ],
                ["most frequent missed (batch, bucket)", missed.most_common(15)],
            ],
        )
    for capture in by_kind.get("graph_capture", []):
        print(
            f"\ngraph capture: {len(capture['shapes'])} shapes, {capture['host_ms'] / 1e3:.1f} s, "
            f"reserved growth {(capture['reserved_after'] - capture['reserved_before']) / 2**30:.2f} GiB"
        )

    hift = by_kind.get("hift", [])
    if hift:
        rows = []
        for low, high in (
            (0, 250),
            (250, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 10**9),
        ):
            calls = [
                h for h in hift if low <= h["history_frames"] + h["new_frames"] < high
            ]
            if calls:
                rows.append(
                    [
                        f"{low}-{high if high < 10**9 else 'max'}",
                        len(calls),
                        spread([c["host_ms"] for c in calls]),
                        spread(
                            [
                                c.get("gpu_ms")
                                for c in calls
                                if c.get("gpu_ms") is not None
                            ]
                        ),
                    ]
                )
        table(
            "HiFT per request call by accumulated frames",
            ["frames", "calls", "host ms", "gpu ms"],
            rows,
        )

    if args.json:
        with open(args.json, "w") as out:
            json.dump(
                {"counts": {k: len(v) for k, v in by_kind.items()}, "warmup": warmup},
                out,
                indent=1,
            )


if __name__ == "__main__":
    main()
