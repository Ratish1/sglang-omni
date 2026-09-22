"""Host stacks inside one NVTX range kind, from an nsys export with CPU sampling.

Takes the CPU samples whose timestamp falls inside a range of the kind on the range's
own thread, then prints the leaf module split, the top leaf symbols and the top
inclusive symbols (a sample counts once for every distinct symbol in its chain), so an
eager stage's host time is attributed to the interpreter, the dispatcher, the library
(cudnn, cublas) or the driver.

usage: python nsys_range_stacks.py REPORT.sqlite --kind pre.spk_encoder [--bench-log LOG] [--top 25]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import sqlite3

from nsys_metrics import bench_window
from ttfc_census import NVTX_PUSH_POP


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--bench-log")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    if args.bench_log:
        t0, t1 = bench_window(db, args.bench_log)
    else:
        t0, t1 = db.execute(
            "select min(start), max(start) from COMPOSITE_EVENTS"
        ).fetchone()
    ranges: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS "
        "where eventType = ? and end is not null and start >= ? and end <= ? order by start",
        (NVTX_PUSH_POP, t0, t1),
    ):
        label = text if text is not None else strings.get(text_id, "")
        words = label.split(" ")
        kind = " ".join(words[:2]) if words[0].startswith("sched.") else words[0]
        if kind == args.kind:
            ranges[tid].append((start, end))
    total_ranges = sum(len(v) for v in ranges.values())
    if not total_ranges:
        raise SystemExit(f"no {args.kind} ranges in the window")
    sample_ids = []
    for tid, spans in ranges.items():
        starts = [s for s, _ in spans]
        for sample_id, ts in db.execute(
            "select id, start from COMPOSITE_EVENTS where globalTid = ? and start between ? and ?",
            (tid, t0, t1),
        ):
            index = bisect.bisect_right(starts, ts) - 1
            if index >= 0 and spans[index][1] >= ts:
                sample_ids.append(sample_id)
    wall_ns = sum(e - s for spans in ranges.values() for s, e in spans)
    print(
        f"{args.kind}: {total_ranges} ranges, {wall_ns / 1e6:.1f} ms wall, "
        f"{len(sample_ids)} samples inside ({len(sample_ids) / total_ranges:.1f} per range)"
    )
    if not sample_ids:
        return
    leaf_module: collections.Counter = collections.Counter()
    leaf_symbol: collections.Counter = collections.Counter()
    inclusive: collections.Counter = collections.Counter()
    chosen = set(sample_ids)
    chains: dict[int, list[tuple[int, str, str]]] = collections.defaultdict(list)
    lo, hi = min(sample_ids), max(sample_ids)
    for sample_id, symbol, module, depth in db.execute(
        "select id, symbol, module, stackDepth from SAMPLING_CALLCHAINS where id between ? and ?",
        (lo, hi),
    ):
        if sample_id in chosen:
            chains[sample_id].append(
                (
                    depth,
                    strings.get(symbol, "?"),
                    strings.get(module, "?").rsplit("/", 1)[-1],
                )
            )
    for chain in chains.values():
        chain.sort()
        _, symbol, module = chain[0]
        leaf_module[module] += 1
        leaf_symbol[f"{symbol[:80]}  [{module}]"] += 1
        for symbol in {f"{s[:80]}  [{m}]" for _, s, m in chain}:
            inclusive[symbol] += 1
    n = len(chains)
    print("\nleaf module share")
    for module, count in leaf_module.most_common(12):
        print(f"  {module:<40}{100 * count / n:>7.1f}%")
    print("\ntop leaf symbols")
    for symbol, count in leaf_symbol.most_common(args.top):
        print(f"  {100 * count / n:>6.1f}%  {symbol}")
    print(
        "\ntop inclusive symbols (share of samples with the symbol anywhere in the chain)"
    )
    for symbol, count in inclusive.most_common(args.top):
        print(f"  {100 * count / n:>6.1f}%  {symbol}")


if __name__ == "__main__":
    main()
