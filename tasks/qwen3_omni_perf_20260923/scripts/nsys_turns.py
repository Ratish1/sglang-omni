"""How two stage processes share one card: kernels of both on one time axis (run on the box).

Reads the nsys sqlite export of a colocated serve (pipeline NVTX on). Within the window it
prints, for the two busiest processes (thinker and talker):
  1. busy time per process, the time both had a kernel in flight (contended: one of them was
     preempted or queued behind the other), and the time only one did;
  2. turns: runs of consecutive kernel starts from one process before the other starts one;
     count, and quantiles of their length in ms and in kernels;
  3. per kernel name of the thinker's decode step, the median span here (to compare with the
     same kernel alone on the card);
  4. per runner/execute range of each process: mean wall, own device union, time under the
     other process's kernels, and host-only remainder;
  5. a text strip of the lane occupancy per millisecond over --strip-ms, and an HTML page with
     the two lanes over --lane-ms (kernels closer than 20 us merged into one bar).

usage: python nsys_turns.py REPORT.sqlite --window window.txt [--strip-ms 200] [--lane-ms 100]
       [--html lanes.html]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import html
import sqlite3
import statistics

from nsys_stage_ledger import nvtx_rows, pid_of, session_window, union_ns

MERGE_NS = 20_000


def intervals_union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def intersection_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    total = 0
    j = 0
    for start, end in a:
        while j < len(b) and b[j][1] <= start:
            j += 1
        k = j
        while k < len(b) and b[k][0] < end:
            total += max(0, min(end, b[k][1]) - max(start, b[k][0]))
            k += 1
    return total


class UnionIndex:
    """A sorted union with its start times, so clipping bisects instead of scanning."""

    def __init__(self, intervals: list[tuple[int, int]]) -> None:
        self.intervals = intervals
        self.starts = [s for s, _ in intervals]


def clipped(union: UnionIndex, t0: int, t1: int) -> list[tuple[int, int]]:
    """Intervals of the union that overlap [t0, t1), clipped to it."""
    first = max(0, bisect.bisect_right(union.starts, t0) - 1)
    out = []
    for s, e in union.intervals[first:]:
        if s >= t1:
            break
        elif e > t0:
            out.append((max(s, t0), min(e, t1)))
        else:
            pass
    return out


def quantiles(values: list[float]) -> str:
    if not values:
        return "none"
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return (
        f"p10 {pick(0.10):.3f} p50 {pick(0.50):.3f} p90 {pick(0.90):.3f} "
        f"p99 {pick(0.99):.3f} max {values[-1]:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    parser.add_argument("--strip-ms", type=int, default=200)
    parser.add_argument("--lane-ms", type=int, default=100)
    parser.add_argument("--html")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    t0, t1 = session_window(db, args.window)
    window_ms = (t1 - t0) / 1e6

    stage_votes: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    execute_ranges: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for start, end, global_tid, annotation in nvtx_rows(db, t0, t1):
        stage = annotation.get("stage", "unknown")
        if annotation.get("kind") == "mark":
            if stage != "unknown":
                stage_votes[pid_of(global_tid)][stage] += 1
        elif end is not None and annotation.get("op") == "execute":
            execute_ranges[pid_of(global_tid)].append((max(start, t0), min(end, t1)))
    stage_by_pid = {
        pid: votes.most_common(1)[0][0] for pid, votes in stage_votes.items()
    }
    names = {i: v for i, v in db.execute("select id, value from StringIds")}
    kernels = db.execute(
        """
        select start, end, globalPid, shortName from CUPTI_ACTIVITY_KIND_KERNEL
        where end > ? and start < ? order by start
        """,
        (t0, t1),
    ).fetchall()
    by_pid: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    for start, end, global_pid, name_id in kernels:
        by_pid[pid_of(global_pid)].append((start, end, names.get(name_id, "?")))
    busiest = sorted(
        by_pid,
        key=lambda pid: -union_ns([(s, e) for s, e, _ in by_pid[pid]]),
    )[:2]
    if len(busiest) < 2:
        raise SystemExit("fewer than two processes launched kernels in the window")
    else:
        pass
    first, second = busiest
    label = {pid: f"{stage_by_pid.get(pid, '?')} ({pid})" for pid in busiest}
    unions = {
        pid: UnionIndex(intervals_union([(s, e) for s, e, _ in by_pid[pid]]))
        for pid in busiest
    }
    contended = intersection_ns(unions[first].intervals, unions[second].intervals)
    print(f"window {window_ms:.1f} ms")
    print("1. busy and contended time")
    for pid in busiest:
        busy = union_ns(unions[pid].intervals)
        print(
            f"  {label[pid]:28s} kernels {len(by_pid[pid]):7d} busy {busy / 1e6:8.1f} ms "
            f"({busy / 1e6 / window_ms:5.1%}) of which under the other's kernels "
            f"{contended / 1e6:8.1f} ms ({contended / max(busy, 1):5.1%})"
        )
    both = union_ns(unions[first].intervals + unions[second].intervals)
    print(f"  card busy (either) {both / 1e6:.1f} ms ({both / 1e6 / window_ms:.1%})")

    print("2. turns (runs of kernel starts from one process)")
    sequence = sorted(
        ((s, e, pid) for pid in busiest for s, e, _ in by_pid[pid]), key=lambda k: k[0]
    )
    turns: dict[int, list[tuple[int, int]]] = {pid: [] for pid in busiest}
    run_pid, run_start, run_end, run_count = None, 0, 0, 0
    for start, end, pid in sequence:
        if pid != run_pid:
            if run_pid is not None:
                turns[run_pid].append((run_end - run_start, run_count))
            else:
                pass
            run_pid, run_start, run_end, run_count = pid, start, end, 0
        else:
            pass
        run_end = max(run_end, end)
        run_count += 1
    if run_pid is not None:
        turns[run_pid].append((run_end - run_start, run_count))
    else:
        pass
    for pid in busiest:
        lengths = [ms / 1e6 for ms, _ in turns[pid]]
        counts = [float(c) for _, c in turns[pid]]
        print(
            f"  {label[pid]:28s} turns {len(turns[pid]):6d} "
            f"({len(turns[pid]) / (window_ms / 1e3):.0f}/s) ms {quantiles(lengths)}"
        )
        print(f"  {'':28s} kernels per turn {quantiles(counts)}")

    print("3. median span per kernel name, top by count, per process (us)")
    for pid in busiest:
        spans: dict[str, list[float]] = collections.defaultdict(list)
        for s, e, name in by_pid[pid]:
            spans[name[:60]].append((e - s) / 1e3)
        print(f"  {label[pid]}")
        for name, values in sorted(spans.items(), key=lambda kv: -len(kv[1]))[:12]:
            print(
                f"    {len(values):7d} x median {statistics.median(values):8.1f} "
                f"p90 {sorted(values)[int(0.9 * len(values))]:8.1f}  {name}"
            )

    print(
        "4. runner/execute ranges: wall, own device, under the other, host-only (ms means)"
    )
    for pid in busiest:
        other = second if pid == first else first
        rows = []
        for start, end in execute_ranges.get(pid, []):
            own = clipped(unions[pid], start, end)
            theirs = clipped(unions[other], start, end)
            own_ns = union_ns(own)
            rows.append(
                (
                    (end - start) / 1e6,
                    own_ns / 1e6,
                    intersection_ns(own, theirs) / 1e6,
                    (union_ns(theirs) - intersection_ns(own, theirs)) / 1e6,
                    ((end - start) - union_ns(own + theirs)) / 1e6,
                )
            )
        if not rows:
            print(f"  {label[pid]:28s} no execute ranges")
            continue
        else:
            pass
        mean = lambda i: statistics.mean(r[i] for r in rows)
        print(
            f"  {label[pid]:28s} ranges {len(rows):6d} wall {mean(0):7.2f} own device "
            f"{mean(1):6.2f} own under other {mean(2):6.2f} other alone {mean(3):6.2f} "
            f"nobody {mean(4):6.2f}"
        )

    print(
        f"5. lane strip, one character per ms over {args.strip_ms} ms from mid window"
    )
    mid = (t0 + t1) // 2
    strip0 = mid
    for pid in busiest:
        chars = []
        for i in range(args.strip_ms):
            a, b = strip0 + i * 1_000_000, strip0 + (i + 1) * 1_000_000
            share = union_ns(clipped(unions[pid], a, b)) / 1e6
            chars.append(" .:-=+*#%@"[min(9, int(share * 10))])
        print(f"  {label[pid]:28s} |{''.join(chars)}|")

    if args.html:
        lane0, lane1 = mid, mid + args.lane_ms * 1_000_000
        width = 1600
        scale = width / (lane1 - lane0)
        parts = []
        colors = {first: "#1f77b4", second: "#d62728"}
        for row, pid in enumerate(busiest):
            bars = []
            for s, e in clipped(unions[pid], lane0, lane1):
                if bars and s - bars[-1][1] < MERGE_NS:
                    bars[-1] = (bars[-1][0], e)
                else:
                    bars.append((s, e))
            y = 20 + row * 60
            parts.append(
                f'<text x="0" y="{y - 4}" font-size="12">{html.escape(label[pid])} '
                f"({len(bars)} bars)</text>"
            )
            for s, e in bars:
                x = (s - lane0) * scale
                w = max(1.0, (e - s) * scale)
                parts.append(
                    f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="40" '
                    f'fill="{colors[pid]}" />'
                )
        ticks = "".join(
            f'<line x1="{i * width / 10:.0f}" y1="0" x2="{i * width / 10:.0f}" y2="150" '
            f'stroke="#bbb" stroke-dasharray="2,4"/><text x="{i * width / 10 + 2:.0f}" '
            f'y="148" font-size="10">{i * args.lane_ms / 10:.0f} ms</text>'
            for i in range(11)
        )
        page = (
            "<!doctype html><html><head><meta charset='utf-8'><title>Card lanes</title>"
            "</head><body style='font-family:sans-serif;background:#fff'>"
            f"<p>{html.escape(args.report)}: {args.lane_ms} ms from the window's middle; "
            "a bar is a run of kernels of one process with gaps under 20 us.</p>"
            f'<svg width="{width}" height="160" viewBox="0 0 {width} 160">{ticks}'
            f"{''.join(parts)}</svg></body></html>"
        )
        with open(args.html, "w") as handle:
            handle.write(page)
        print(f"lanes written to {args.html}")
    else:
        pass


if __name__ == "__main__":
    main()
