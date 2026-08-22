"""Attribute GPU idle in an nsys capture to OmniScheduler host phases.

Input: the sqlite export of a capture taken with profile/nvtx_probe
installed (nsys export --type sqlite <rep>). Output: markdown tables on
stdout and an optional JSON dump.

    python profile/nsys_gap_attribution.py capture.sqlite [--threshold-us 100]
        [--window exec|recv|START_NS END_NS] [--json out.json]

Definitions:
  window      exec (default): first to last exec:* range on the scheduler
              thread, the benchmark pass itself; recv: first to last
              sched:recv, the whole capture; or explicit ns bounds
  GPU busy    union of kernel, graph, memcpy and memset intervals on every
              device, stream and process, clipped to the window
  idle        window minus busy, as a set of gaps
  big gap     a gap at or above the threshold; attributed time-weighted:
              each part of it goes to the innermost scheduler-thread label
              active at that instant, so a gap that spans several phases
              is split across them and a gap with no label is unlabeled
  micro gap   a gap below the threshold (a bubble between launches inside
              a step); attributed to the innermost label at its start and
              to the enclosing exec:* label, if any
  iteration   the interval between consecutive sched:recv starts
Labels are aggregated by base name (the text before the first space), so
exec:launch:decode bs=16 and bs=17 are one row; the extend shape has its
own histogram.
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


def base(name: str) -> str:
    return name.split(" ", 1)[0]


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
        n = 0
        for start, end in con.execute(
            f"select start, end from {table} where end is not null"
        ):
            out.append((start, end))
            n += 1
        counts[table.replace("CUPTI_ACTIVITY_KIND_", "")] = n
    out.sort()
    return out, counts


def merge(intervals):
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


def overlap_total(a, b):
    """Total overlap between two sorted, merged interval lists."""
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


class LabelTimeline:
    """Piecewise-constant innermost label (and enclosing exec label) of one thread.

    Built by a sweep over range boundaries. segment k covers
    [bounds[k], bounds[k+1]) with labels inner[k] and execl[k] (None if no
    range is active).
    """

    def __init__(self, ranges):
        events = []
        for i, (s, e, _n, _t) in enumerate(ranges):
            events.append((s, 1, i))
            events.append((e, 0, i))
        events.sort()
        self.bounds, self.inner, self.execl = [], [], []
        active: dict[int, tuple[int, str]] = {}
        k = 0
        while k < len(events):
            t = events[k][0]
            while k < len(events) and events[k][0] == t:
                _t, kind, i = events[k]
                if kind == 1:
                    active[i] = (ranges[i][0], ranges[i][2])
                else:
                    active.pop(i, None)
                k += 1
            if active:
                inner = max(active.values())[1]
                execs = [v for v in active.values() if v[1].startswith("exec:")]
                execl = max(execs)[1] if execs else None
            else:
                inner = execl = None
            self.bounds.append(t)
            self.inner.append(inner)
            self.execl.append(execl)

    def at(self, t):
        k = bisect.bisect_right(self.bounds, t) - 1
        if k < 0:
            return None, None
        return self.inner[k], self.execl[k]

    def split(self, s, e):
        """Yield (label, ns) pieces of [s, e) by innermost label."""
        k = bisect.bisect_right(self.bounds, s) - 1
        cursor = s
        while cursor < e:
            nxt = self.bounds[k + 1] if k + 1 < len(self.bounds) else e
            piece_end = min(e, nxt)
            label = self.inner[k] if k >= 0 else None
            yield label, piece_end - cursor
            cursor = piece_end
            k += 1


def pct(vals, p):
    if not vals:
        return 0.0
    vs = sorted(vals)
    return vs[min(len(vs) - 1, max(0, math.ceil(p * len(vs)) - 1))]


def ms(ns: float) -> float:
    return ns / 1e6


def stats_row(vals):
    return {
        "count": len(vals),
        "total_ms": ms(sum(vals)),
        "median_ms": ms(statistics.median(vals)) if vals else 0.0,
        "p90_ms": ms(pct(vals, 0.9)),
    }


def table(title, header, rows):
    print(f"## {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--threshold-us", type=float, default=100.0)
    ap.add_argument(
        "--window", nargs="+", default=["exec"], metavar="exec|recv|START END"
    )
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
    build_tids = [
        tid
        for tid, rs in by_tid.items()
        if any(r[2] == "build:req" for r in rs) and tid != sched_tid
    ]
    sched = by_tid[sched_tid]

    execs = [r for r in sched if r[2].startswith("exec:")]
    if args.window == ["exec"]:
        if not execs:
            sys.exit(
                "no exec:* range on the scheduler thread: nothing ran in the capture"
            )
        lo, hi = min(r[0] for r in execs), max(r[1] for r in execs)
    elif args.window == ["recv"]:
        recvs = [r for r in sched if r[2] == "sched:recv"]
        lo, hi = recvs[0][0], max(r[1] for r in recvs)
    elif len(args.window) == 2:
        lo, hi = int(args.window[0]), int(args.window[1])
    else:
        sys.exit("--window takes exec, recv, or START END in ns")
    window_ns = hi - lo

    busy = merge(clip(gpu, lo, hi))
    busy_ns = sum(e - s for s, e in busy)
    idle_ns = window_ns - busy_ns
    gaps = gaps_between(busy, lo, hi)
    thr_ns = int(args.threshold_us * 1e3)
    big = [(s, e) for s, e in gaps if e - s >= thr_ns]
    micro = [(s, e) for s, e in gaps if e - s < thr_ns]

    sched_in = [r for r in sched if r[1] > lo and r[0] < hi]
    timeline = LabelTimeline(sched_in)

    big_attr = defaultdict(int)
    big_gap_count = defaultdict(int)
    big_gap_sizes = defaultdict(list)
    for s, e in big:
        seen = set()
        for label, dur in timeline.split(s, e):
            key = base(label) if label else "unlabeled"
            big_attr[key] += dur
            seen.add(key)
        for key in seen:
            big_gap_count[key] += 1
            big_gap_sizes[key].append(e - s)

    micro_inner = defaultdict(int)
    micro_exec = defaultdict(int)
    micro_count = defaultdict(int)
    for s, e in micro:
        inner, execl = timeline.at(s)
        micro_inner[base(inner) if inner else "unlabeled"] += e - s
        key = base(execl) if execl else "outside exec"
        micro_exec[key] += e - s
        micro_count[key] += 1
    micro_ns = sum(e - s for s, e in micro)

    recv_in = sorted(r[0] for r in sched if r[2] == "sched:recv" and lo <= r[0] < hi)

    def iteration_index(t):
        return bisect.bisect_right(recv_in, t) - 1

    hold_times = sorted(
        t
        for t, name, tid in marks
        if tid == sched_tid and name.startswith("sched:hold") and lo <= t < hi
    )
    hold_iters = {iteration_index(t) for t in hold_times}
    idle_in_hold = sum(e - s for s, e in big if iteration_index(s) in hold_iters)

    host = defaultdict(list)
    for s, e, name, _tid in sched_in:
        host[base(name)].append(min(e, hi) - max(s, lo))
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

    exec_union = merge(
        clip([(s, e) for s, e, n, _t in sched_in if n.startswith("exec:")], lo, hi)
    )
    big_merged = merge(big)

    def other_thread_block(tids, prefix):
        rows = {}
        if not tids:
            return rows, 0, 0
        rs = [r for t in tids for r in by_tid[t] if r[1] > lo and r[0] < hi]
        per = defaultdict(list)
        for s, e, name, _tid in rs:
            per[base(name)].append(min(e, hi) - max(s, lo))
        rows = {k: stats_row(v) for k, v in sorted(per.items())}
        active = merge(clip([(s, e) for s, e, _n, _t in rs], lo, hi))
        return (
            rows,
            overlap_total(active, exec_union),
            overlap_total(active, big_merged),
        )

    enc_rows, enc_vs_exec, enc_vs_idle = other_thread_block(enc_tids, "enc:")
    build_rows, build_vs_exec, build_vs_idle = other_thread_block(build_tids, "build:")
    enc_sync = merge(
        clip(
            [(s, e) for t in enc_tids for s, e, n, _t in by_tid[t] if n == "enc:sync"],
            lo,
            hi,
        )
    )
    enc_sync_vs_exec = overlap_total(enc_sync, exec_union) if enc_tids else 0

    iter_walls = [b - a for a, b in zip(recv_in, recv_in[1:])]
    n_iters = max(len(recv_in) - 1, 0)
    iters_with_exec = {
        i
        for i in (iteration_index(r[0]) for r in execs if lo <= r[0] < hi)
        if 0 <= i < n_iters
    }
    no_exec_share = 1 - len(iters_with_exec) / n_iters if n_iters else 0.0

    out = {
        "gpu_tables": gpu_counts,
        "window_mode": " ".join(args.window),
        "window_ns": [lo, hi],
        "window_ms": ms(window_ns),
        "gpu_busy_ms": ms(busy_ns),
        "gpu_idle_ms": ms(idle_ns),
        "gpu_idle_pct": 100 * idle_ns / window_ns if window_ns else 0,
        "threshold_us": args.threshold_us,
        "gaps_total": len(gaps),
        "big_gaps": len(big),
        "micro_gaps": len(micro),
        "micro_idle_ms": ms(micro_ns),
        "micro_idle_pct_of_idle": 100 * micro_ns / idle_ns if idle_ns else 0,
        "scheduler_threads_found": len(sched_tids),
        "encoder_threads_found": len(enc_tids),
        "builder_threads_found": len(build_tids),
        "big_attribution": {
            k: {
                "gaps": big_gap_count[k],
                "idle_ms": ms(v),
                "pct_of_idle": 100 * v / idle_ns if idle_ns else 0,
                "median_gap_ms": ms(statistics.median(big_gap_sizes[k])),
                "p90_gap_ms": ms(pct(big_gap_sizes[k], 0.9)),
            }
            for k, v in big_attr.items()
        },
        "micro_by_exec": {
            k: {
                "gaps": micro_count[k],
                "idle_ms": ms(v),
                "pct_of_idle": 100 * v / idle_ns if idle_ns else 0,
            }
            for k, v in micro_exec.items()
        },
        "micro_by_inner": {k: ms(v) for k, v in micro_inner.items()},
        "host": {k: stats_row(v) for k, v in host.items()},
        "host_unlabeled_ms": ms(unlabeled_host_ns),
        "extend_bs_hist": dict(
            sorted((str(k), extend_bs.count(k)) for k in set(extend_bs))
        ),
        "extend_tok_median": statistics.median(extend_tok) if extend_tok else None,
        "extend_tok_p90": pct(extend_tok, 0.9) if extend_tok else None,
        "hold_marks": len(hold_times),
        "hold_iterations": len(hold_iters),
        "idle_in_hold_iterations_ms": ms(idle_in_hold),
        "encoder": enc_rows,
        "encoder_overlap_exec_ms": ms(enc_vs_exec),
        "encoder_overlap_idle_ms": ms(enc_vs_idle),
        "enc_sync_overlap_exec_ms": ms(enc_sync_vs_exec),
        "builder": build_rows,
        "builder_overlap_exec_ms": ms(build_vs_exec),
        "builder_overlap_idle_ms": ms(build_vs_idle),
        "iterations": n_iters,
        "iteration_median_ms": (
            ms(statistics.median(iter_walls)) if iter_walls else None
        ),
        "iteration_p90_ms": ms(pct(iter_walls, 0.9)) if iter_walls else None,
        "iterations_without_exec_pct": 100 * no_exec_share,
    }

    print(
        f"## Window ({out['window_mode']}): {ms(window_ns):.1f} ms  ({lo} .. {hi} ns); threads: scheduler {len(sched_tids)}, encoder {len(enc_tids)}, builder {len(build_tids)}; GPU rows {gpu_counts}"
    )
    print(
        f"GPU busy {out['gpu_busy_ms']:.1f} ms, idle {out['gpu_idle_ms']:.1f} ms ({out['gpu_idle_pct']:.1f}%); gaps {len(gaps)}: {len(big)} at or above {args.threshold_us:.0f} us, {len(micro)} micro holding {out['micro_idle_ms']:.1f} ms ({out['micro_idle_pct_of_idle']:.1f}% of idle)\n"
    )
    table(
        "Big-gap attribution, time-weighted by innermost scheduler-thread label",
        [
            "label",
            "gaps touched",
            "idle ms",
            "% of idle",
            "median gap ms",
            "p90 gap ms",
        ],
        [
            (
                k,
                v["gaps"],
                f"{v['idle_ms']:.1f}",
                f"{v['pct_of_idle']:.1f}",
                f"{v['median_gap_ms']:.3f}",
                f"{v['p90_gap_ms']:.3f}",
            )
            for k, v in sorted(
                out["big_attribution"].items(), key=lambda kv: -kv[1]["idle_ms"]
            )
        ],
    )
    print(
        f"Idle inside hold iterations: {out['idle_in_hold_iterations_ms']:.1f} ms over {len(hold_iters)} iterations ({len(hold_times)} hold marks)\n"
    )
    table(
        "Micro-gap idle by enclosing exec label",
        ["enclosing exec", "gaps", "idle ms", "% of idle"],
        [
            (k, v["gaps"], f"{v['idle_ms']:.1f}", f"{v['pct_of_idle']:.1f}")
            for k, v in sorted(
                out["micro_by_exec"].items(), key=lambda kv: -kv[1]["idle_ms"]
            )
        ],
    )
    table(
        "Micro-gap idle by innermost label",
        ["innermost", "idle ms"],
        [
            (k, f"{v:.1f}")
            for k, v in sorted(out["micro_by_inner"].items(), key=lambda kv: -kv[1])
        ],
    )
    table(
        "Host phase totals (scheduler thread)",
        ["label", "count", "total ms", "median ms", "p90 ms"],
        [
            (
                k,
                v["count"],
                f"{v['total_ms']:.1f}",
                f"{v['median_ms']:.3f}",
                f"{v['p90_ms']:.3f}",
            )
            for k, v in sorted(out["host"].items(), key=lambda kv: -kv[1]["total_ms"])
        ]
        + [("unlabeled", "", f"{out['host_unlabeled_ms']:.1f}", "", "")],
    )
    print(
        f"## Extend batch shape: bs histogram {out['extend_bs_hist']}; tok median {out['extend_tok_median']}, p90 {out['extend_tok_p90']}\n"
    )
    for name, rows, vs_exec, vs_idle in (
        ("Encoder thread", enc_rows, enc_vs_exec, enc_vs_idle),
        ("Builder thread", build_rows, build_vs_exec, build_vs_idle),
    ):
        if rows:
            table(
                name,
                ["label", "count", "total ms", "median ms", "p90 ms"],
                [
                    (
                        k,
                        v["count"],
                        f"{v['total_ms']:.1f}",
                        f"{v['median_ms']:.3f}",
                        f"{v['p90_ms']:.3f}",
                    )
                    for k, v in rows.items()
                ],
            )
            print(
                f"{name} active during scheduler exec: {ms(vs_exec):.1f} ms; during big idle gaps: {ms(vs_idle):.1f} ms"
                + (
                    f"; enc:sync during exec: {ms(enc_sync_vs_exec):.1f} ms"
                    if name.startswith("Encoder")
                    else ""
                )
                + "\n"
            )
    print(
        f"## Iterations: {n_iters}; wall median {out['iteration_median_ms']} ms, p90 {out['iteration_p90_ms']} ms; without any exec range: {out['iterations_without_exec_pct']:.1f}%"
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
