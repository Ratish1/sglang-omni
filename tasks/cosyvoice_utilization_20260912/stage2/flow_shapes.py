#!/usr/bin/env python3
"""Aggregate the Flow solve shapes a server logged into the numbers a graph
shape key needs.

Reads one or more serve logs carrying the `Fun-CosyVoice3 Flow solve` lines and
reports, per path, how many calls each shape accounted for and what a bucket
rule would have to cover. The point is to derive the key from the calls a
deployment makes rather than from a table traced on one card.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

LINE = re.compile(
    r"Fun-CosyVoice3 Flow solve path=(\w+) rows=(\d+) width=(\d+) total=(\d+) "
    r"lengths=([\d,]*)"
)


def solves(paths: list[Path]) -> list[tuple[str, int, int, int, tuple[int, ...]]]:
    found = []
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            match = LINE.search(line)
            if match is None:
                continue
            found.append(
                (
                    match.group(1),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                    tuple(int(v) for v in match.group(5).split(",") if v),
                )
            )
    return found


def quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def geometric_buckets(quantum: int, ceiling: int, waste: float) -> list[int]:
    """Bucket edges whose round-up padding never exceeds `waste` of the call.

    The quantum is the attention chunk in frames: no Flow call length is
    anything but a multiple of it, so nothing finer can occur. The ratio
    follows from the padding bound alone, and the ceiling is the scheduler's
    admission budget. No measurement enters this.
    """
    if not 0 < waste < 1:
        raise ValueError(f"waste must be a fraction in (0, 1), got {waste}")
    edges = [quantum]
    while edges[-1] < ceiling:
        nxt = int(-(-int(edges[-1] / (1 - waste)) // quantum) * quantum)
        edges.append(min(max(nxt, edges[-1] + quantum), ceiling))
    return edges


def report(records, quantum: int, ceiling: int, waste: float) -> None:
    by_path = Counter(record[0] for record in records)
    print(f"{len(records)} Flow solves")
    for path, count in by_path.most_common():
        print(f"  {path:13s} {count:6d}  {count / len(records):6.1%}")

    graphed = by_path.get("graph", 0)
    padded = by_path.get("padded", 0)
    if graphed or padded:
        print(
            f"\ngraph hit rate over the padded path: {graphed}/{graphed + padded} "
            f"= {graphed / (graphed + padded):.1%}"
        )

    print("\nper path shape spread")
    print(f"  {'path':13s} {'calls':>6s} {'rows':>9s} {'width p50':>10s} "
          f"{'width max':>10s} {'total p50':>10s} {'total max':>10s} {'keys':>6s}")
    for path in by_path:
        rows = [r for r in records if r[0] == path]
        widths = [r[2] for r in rows]
        totals = [r[3] for r in rows]
        keys = len({(r[1], r[2]) for r in rows})
        print(
            f"  {path:13s} {len(rows):6d} {min(r[1] for r in rows):4d}"
            f"-{max(r[1] for r in rows):<4d} {quantile(widths, 0.5):10d} "
            f"{max(widths):10d} {quantile(totals, 0.5):10d} {max(totals):10d} "
            f"{keys:6d}"
        )

    off_quantum = [r for r in records if any(length % quantum for length in r[4])]
    print(
        f"\nrow lengths off the {quantum} frame quantum: "
        f"{len(off_quantum)} of {len(records)}"
    )

    edges = geometric_buckets(quantum, ceiling, waste)
    print(
        f"\nderived buckets, quantum {quantum}, ceiling {ceiling}, "
        f"padding bound {waste:.0%}: {len(edges)} of them"
    )
    print("  " + " ".join(str(edge) for edge in edges))
    for path in ("packed_hop", "packed_final", "padded", "graph"):
        totals = [r[3] for r in records if r[0] == path]
        if not totals:
            continue
        waste_frames = 0
        for total in totals:
            edge = next((e for e in edges if e >= total), edges[-1])
            waste_frames += max(edge - total, 0)
        print(
            f"  {path:13s} {len(totals):5d} calls, "
            f"{sum(totals):8d} real frames, "
            f"{waste_frames:7d} padded ({waste_frames / sum(totals):.1%}), "
            f"{sum(1 for t in totals if t > edges[-1]):4d} above the ceiling"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument(
        "--quantum",
        type=int,
        default=50,
        help="attention chunk in mel frames, flow.static_chunk_size",
    )
    parser.add_argument(
        "--ceiling",
        type=int,
        default=8000,
        help="frames one call can reach, flow_batch_admission_frames",
    )
    parser.add_argument(
        "--waste",
        type=float,
        default=0.25,
        help="padding a round-up may add, as the merge path's pad budget does",
    )
    args = parser.parse_args()
    records = solves(args.logs)
    if not records:
        raise SystemExit("no Flow solve lines in those logs")
    report(records, args.quantum, args.ceiling, args.waste)


if __name__ == "__main__":
    main()
