"""Attribute GPU idle gaps in an nsys capture to OmniScheduler host phases.

Input: the sqlite export of a capture taken with tasks/nvtx_probe installed
(nsys export --type sqlite <rep>). Output: the seven tables of
tasks/loop_profile_plan_20260822.md section 5, as markdown on stdout, plus
an optional JSON dump.

    python tasks/nsys_gap_attribution.py capture.sqlite [--threshold-us 100]
        [--window START_NS END_NS] [--trim 0.10] [--json out.json]

Definitions:
  window      first to last sched:recv on the scheduler thread, trimmed by
              --trim (default 0: the capture spans complete benchmark
              passes) at each end, unless --window is given (ns)
  GPU busy    union of kernel, memcpy and memset intervals on every device,
              stream and process, clipped to the window
  idle gap    a maximal sub-interval of the window with no GPU busy time
  attribution the innermost scheduler-thread NVTX range active at the gap
              start; none active is reported as unlabeled
  iteration   the interval between consecutive sched:recv starts
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sqlite3
import statistics
import sys
from collections import defaultdict

PUSHPOP = "NvtxPushPopRange"
STARTEND = "NvtxStartEndRange"
MARK = "NvtxMark"
FALLBACK_EVENT_TYPES = {PUSHPOP: 59, STARTEND: 60, MARK: 34}
GPU_TABLES = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
)
EXTEND_RE = re.compile(r"^exec:(sync|launch):extend bs=(\d+) tok=(\d+)$")


def tables(con) -> set[str]:
    return {
        r[0] for r in con.execute("select name from sqlite_master where type='table'")
    }


def columns(con, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"pragma table_info({table})")}


def event_type_ids(con, present: set[str]) -> dict[str, int]:
    ids = dict(FALLBACK_EVENT_TYPES)
    if "ENUM_NVTX_EVENT_TYPE" in present:
        for eid, name in con.execute("select id, name from ENUM_NVTX_EVENT_TYPE"):
            if name in ids:
                ids[name] = eid
    return ids


def load_nvtx(con, present: set[str]):
    """Return (ranges, marks): ranges as (start, end, name, tid), marks as (t, name, tid)."""
    if "NVTX_EVENTS" not in present:
        sys.exit("no NVTX_EVENTS table: was the probe installed and --trace=nvtx set?")
    cols = columns(con, "NVTX_EVENTS")
    ids = event_type_ids(con, present)
    text_expr = "coalesce(e.text, s.value)" if "textId" in cols else "e.text"
    join = "left join StringIds s on s.id = e.textId" if "textId" in cols else ""
    rows = con.execute(
        f"select e.start, e.end, e.eventType, e.globalTid, {text_expr} from NVTX_EVENTS e {join}"
    )
    ranges, marks = [], []
    for start, end, etype, gtid, name in rows:
        if name is None:
            continue
        if etype == ids[MARK] or end is None:
            marks.append((start, name, gtid))
        elif etype in (ids[PUSHPOP], ids[STARTEND]):
            ranges.append((start, end, name, gtid))
    ranges.sort()
    marks.sort()
    return ranges, marks


def load_gpu_intervals(con, present: set[str]):
    """GPU work intervals and the per-table row counts they came from.

    With --cuda-graph-trace=graph each graph replay is one GRAPH_TRACE row
    and its kernels are absent from KERNEL; with node it is the reverse.
    Both are read, so either capture mode gives the full busy union.
    """
    out, counts = [], {}
    for table in GPU_TABLES:
        if table not in present:
            continue
        cols = columns(con, table)
        dev = "deviceId" if "deviceId" in cols else "0"
        n = 0
        for start, end, device in con.execute(
            f"select start, end, {dev} from {table} where end is not null"
        ):
            out.append((start, end, device))
            n += 1
        counts[table.replace("CUPTI_ACTIVITY_KIND_", "")] = n
    out.sort()
    return out, counts


def merge(intervals):
    """Union of (start, end) pairs, sorted by start."""
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def clip(intervals, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi]


def gaps_between(busy, lo, hi):
    gaps, cursor = [], lo
    for s, e in busy:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < hi:
        gaps.append((cursor, hi))
    return gaps


def innermost_at(times, ranges):
    """For each query time, the innermost range active there (sweep). Returns names list."""
    events = []
    for i, (s, e, name, _tid) in enumerate(ranges):
        events.append((s, 1, i))
        events.append((e, 0, i))
    queries = sorted((t, 2, qi) for qi, t in enumerate(times))
    events.sort()
    out = [None] * len(times)
    active: dict[int, tuple[int, str]] = {}
    ei = 0
    for t, _kind, qi in queries:
        while ei < len(events) and (
            events[ei][0] < t or (events[ei][0] == t and events[ei][1] == 1)
        ):
            _et, ekind, i = events[ei]
            if ekind == 1:
                active[i] = (ranges[i][0], ranges[i][2])
            else:
                active.pop(i, None)
            ei += 1
        if active:
            out[qi] = max(active.values())[1]
    return out


def pct(vals, p):
    if not vals:
        return 0.0
    vs = sorted(vals)
    return vs[min(len(vs) - 1, max(0, math.ceil(p * len(vs)) - 1))]


def ms(ns: float) -> float:
    return ns / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--threshold-us", type=float, default=100.0)
    ap.add_argument("--window", type=int, nargs=2, metavar=("START_NS", "END_NS"))
    ap.add_argument("--trim", type=float, default=0.0)
    ap.add_argument("--json")
    args = ap.parse_args()

    con = sqlite3.connect(args.sqlite)
    present = tables(con)
    ranges, marks = load_nvtx(con, present)
    gpu, gpu_counts = load_gpu_intervals(con, present)
    if not gpu:
        sys.exit("no GPU activity rows: check --trace=cuda and the capture window")

    by_tid = defaultdict(list)
    for r in ranges:
        by_tid[r[3]].append(r)
    sched_tids = [
        tid for tid, rs in by_tid.items() if any(r[2] == "sched:recv" for r in rs)
    ]
    if not sched_tids:
        sys.exit("no sched:recv range found: probe not installed on a scheduler thread")
    sched_tid = max(
        sched_tids, key=lambda t: sum(r[2].startswith("exec:") for r in by_tid[t])
    )
    enc_tids = [
        tid for tid, rs in by_tid.items() if any(r[2].startswith("enc:") for r in rs)
    ]
    sched = by_tid[sched_tid]
    recv_starts = [r[0] for r in sched if r[2] == "sched:recv"]

    if args.window:
        lo, hi = args.window
    else:
        first, last = recv_starts[0], max(r[1] for r in sched if r[2] == "sched:recv")
        span = last - first
        lo, hi = int(first + args.trim * span), int(last - args.trim * span)
    window_ns = hi - lo

    busy = merge(clip([(s, e) for s, e, _d in gpu], lo, hi))
    busy_ns = sum(e - s for s, e in busy)
    idle_ns = window_ns - busy_ns
    gaps = gaps_between(busy, lo, hi)
    thr_ns = int(args.threshold_us * 1e3)
    big = [(s, e) for s, e in gaps if e - s >= thr_ns]
    small_ns = sum(e - s for s, e in gaps) - sum(e - s for s, e in big)

    sched_in = [r for r in sched if r[1] > lo and r[0] < hi]
    labels = innermost_at([s for s, _e in big], sched_in)
    attrib = defaultdict(list)
    for (s, e), name in zip(big, labels):
        attrib[name or "unlabeled"].append(e - s)

    hold_times = sorted(
        t
        for t, name, tid in marks
        if tid == sched_tid and name.startswith("sched:hold") and lo <= t < hi
    )
    recv_in = sorted(t for t in recv_starts if lo <= t < hi)

    def iteration_index(t):
        return bisect.bisect_right(recv_in, t) - 1

    hold_iters = {iteration_index(t) for t in hold_times}
    idle_in_hold_iters = sum(e - s for s, e in big if iteration_index(s) in hold_iters)

    host = defaultdict(list)
    for s, e, name, _tid in sched_in:
        base = name.split(" ")[0]
        host[base].append(min(e, hi) - max(s, lo))
    covered = sum(
        e - s for s, e in merge(clip([(s, e) for s, e, _n, _t in sched_in], lo, hi))
    )
    unlabeled_host_ns = window_ns - covered

    extend_bs, extend_tok = [], []
    for s, e, name, _tid in sched_in:
        m = EXTEND_RE.match(name)
        if m:
            extend_bs.append(int(m.group(2)))
            extend_tok.append(int(m.group(3)))

    enc_rows = []
    enc_overlap_gap_ns = 0
    enc_sync_overlap_exec_ns = 0
    if enc_tids:
        enc = [r for t in enc_tids for r in by_tid[t] if r[1] > lo and r[0] < hi]
        enc_active = merge(clip([(s, e) for s, e, _n, _t in enc], lo, hi))
        for s, e in big:
            for a, b in enc_active:
                if a < e and s < b:
                    enc_overlap_gap_ns += min(e, b) - max(s, a)
        exec_union = merge(
            clip([(s, e) for s, e, n, _t in sched_in if n.startswith("exec:")], lo, hi)
        )
        sync = merge(clip([(s, e) for s, e, n, _t in enc if n == "enc:sync"], lo, hi))
        for s, e in sync:
            for a, b in exec_union:
                if a < e and s < b:
                    enc_sync_overlap_exec_ns += min(e, b) - max(s, a)
        enc_host = defaultdict(list)
        for s, e, name, _tid in enc:
            enc_host[name.split(" ")[0]].append(e - s)
        enc_rows = [
            (k, len(v), ms(sum(v)), ms(statistics.median(v)), ms(pct(v, 0.9)))
            for k, v in sorted(enc_host.items())
        ]

    iter_walls = [b - a for a, b in zip(recv_in, recv_in[1:])]
    exec_starts = sorted(s for s, _e, n, _t in sched_in if n.startswith("exec:"))
    iters_with_exec = {
        i
        for i in (iteration_index(t) for t in exec_starts)
        if 0 <= i < max(len(recv_in) - 1, 0)
    }
    n_iters = max(len(recv_in) - 1, 0)
    no_exec_share = 1 - len(iters_with_exec) / n_iters if n_iters else 0.0

    out = {
        "gpu_tables": gpu_counts,
        "window_ns": [lo, hi],
        "window_ms": ms(window_ns),
        "gpu_busy_ms": ms(busy_ns),
        "gpu_idle_ms": ms(idle_ns),
        "gpu_idle_pct": 100 * idle_ns / window_ns if window_ns else 0,
        "threshold_us": args.threshold_us,
        "gaps_total": len(gaps),
        "gaps_at_or_above_threshold": len(big),
        "idle_below_threshold_ms": ms(small_ns),
        "scheduler_threads_found": len(sched_tids),
        "encoder_threads_found": len(enc_tids),
        "attribution": {
            k: {
                "count": len(v),
                "idle_ms": ms(sum(v)),
                "pct_of_idle": 100 * sum(v) / idle_ns if idle_ns else 0,
                "median_ms": ms(statistics.median(v)),
                "p90_ms": ms(pct(v, 0.9)),
            }
            for k, v in attrib.items()
        },
        "host": {
            k: {
                "count": len(v),
                "total_ms": ms(sum(v)),
                "median_ms": ms(statistics.median(v)),
                "p90_ms": ms(pct(v, 0.9)),
            }
            for k, v in host.items()
        },
        "host_unlabeled_ms": ms(unlabeled_host_ns),
        "extend_bs_hist": dict(
            sorted((str(k), extend_bs.count(k)) for k in set(extend_bs))
        ),
        "extend_tok_median": statistics.median(extend_tok) if extend_tok else None,
        "extend_tok_p90": pct(extend_tok, 0.9) if extend_tok else None,
        "hold_marks": len(hold_times),
        "hold_iterations": len(hold_iters),
        "idle_in_hold_iterations_ms": ms(idle_in_hold_iters),
        "encoder": enc_rows,
        "idle_overlapping_encoder_ms": ms(enc_overlap_gap_ns),
        "enc_sync_overlapping_exec_ms": ms(enc_sync_overlap_exec_ns),
        "iterations": n_iters,
        "iteration_median_ms": (
            ms(statistics.median(iter_walls)) if iter_walls else None
        ),
        "iteration_p90_ms": ms(pct(iter_walls, 0.9)) if iter_walls else None,
        "iterations_without_exec_pct": 100 * no_exec_share,
    }

    print(
        f"## Window: {ms(window_ns):.1f} ms  ({lo} .. {hi} ns); scheduler threads {len(sched_tids)}, encoder threads {len(enc_tids)}"
    )
    print(
        f"GPU busy {out['gpu_busy_ms']:.1f} ms, idle {out['gpu_idle_ms']:.1f} ms ({out['gpu_idle_pct']:.1f}%); gaps {len(gaps)}, at or above {args.threshold_us:.0f} us: {len(big)}; idle below threshold {out['idle_below_threshold_ms']:.1f} ms\n"
    )
    print("## Gap attribution (innermost scheduler-thread label at gap start)\n")
    print(
        "| label | gaps | idle ms | % of idle | median ms | p90 ms |\n|---|---:|---:|---:|---:|---:|"
    )
    for k, v in sorted(out["attribution"].items(), key=lambda kv: -kv[1]["idle_ms"]):
        print(
            f"| {k} | {v['count']} | {v['idle_ms']:.1f} | {v['pct_of_idle']:.1f} | {v['median_ms']:.3f} | {v['p90_ms']:.3f} |"
        )
    print(
        f"\nIdle inside hold iterations: {out['idle_in_hold_iterations_ms']:.1f} ms over {len(hold_iters)} iterations ({len(hold_times)} hold marks)\n"
    )
    print("## Host phase totals (scheduler thread)\n")
    print(
        "| label | count | total ms | median ms | p90 ms |\n|---|---:|---:|---:|---:|"
    )
    for k, v in sorted(out["host"].items(), key=lambda kv: -kv[1]["total_ms"]):
        print(
            f"| {k} | {v['count']} | {v['total_ms']:.1f} | {v['median_ms']:.3f} | {v['p90_ms']:.3f} |"
        )
    print(f"| unlabeled | | {out['host_unlabeled_ms']:.1f} | | |\n")
    print(
        f"## Extend batch shape: bs histogram {out['extend_bs_hist']}; tok median {out['extend_tok_median']}, p90 {out['extend_tok_p90']}\n"
    )
    if enc_rows:
        print(
            "## Encoder thread\n\n| label | count | total ms | median ms | p90 ms |\n|---|---:|---:|---:|---:|"
        )
        for k, n, tot, med, p90 in enc_rows:
            print(f"| {k} | {n} | {tot:.1f} | {med:.3f} | {p90:.3f} |")
        print(
            f"\nIdle overlapping an encoder range: {out['idle_overlapping_encoder_ms']:.1f} ms; enc:sync overlapping scheduler exec: {out['enc_sync_overlapping_exec_ms']:.1f} ms\n"
        )
    print(
        f"## Iterations: {n_iters}; wall median {out['iteration_median_ms']} ms, p90 {out['iteration_p90_ms']} ms; without any exec range: {out['iterations_without_exec_pct']:.1f}%"
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
