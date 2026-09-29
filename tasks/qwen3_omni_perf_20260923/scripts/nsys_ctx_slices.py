"""GPU context residency of a colocated serve (run on the box), from an nsys export taken
with --gpuctxsw=true and the pipeline NVTX annotations on.

Pairs each context's RESTORE_START with its next SAVE_END into residency slices, names each
context by the stage its NVTX annotations carry, and prints within the window:
  1. per stage: slices, resident time and share of the window, slice length quantiles, how
     its slices ended (the front-end ack before the save: WFI, CTAP, CILP), and resident time
     with none of its own kernels running;
  2. the switch cost: the gap from one context's SAVE_END to the next RESTORE_START;
  3. per runner/execute range of each stage, by forward mode: wall, own residency, the other
     stages' residency, switches in, own kernel time;
  4. the time with no context resident.
usage: python nsys_ctx_slices.py REPORT.sqlite [--window window.txt]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import sqlite3
import statistics

from nsys_stage_ledger import nvtx_rows, pid_of, session_window

RESTORE_START = 8
SAVE_END = 7
ACK_TAGS = {2: "ACK", 3: "WFI", 4: "GFXP", 5: "CTAP", 6: "CILP"}


def overlap_ns(
    intervals: list[tuple[int, int]], starts: list[int], t0: int, t1: int
) -> int:
    total = 0
    index = max(0, bisect.bisect_right(starts, t0) - 1)
    while index < len(intervals) and intervals[index][0] < t1:
        start, end = intervals[index]
        total += max(0, min(end, t1) - max(start, t0))
        index += 1
    return total


def merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def quantiles(values: list[float]) -> str:
    if not values:
        return "none"
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return f"p10 {pick(0.1):.3f} p50 {pick(0.5):.3f} p90 {pick(0.9):.3f} p99 {pick(0.99):.3f} max {ordered[-1]:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    t0, t1 = session_window(db, args.window)
    window_ns = t1 - t0

    stage_votes: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    execute_ranges: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(
        list
    )
    for start, end, global_tid, annotation in nvtx_rows(db, t0, t1):
        stage = annotation.get("stage", "unknown")
        if annotation.get("kind") == "mark":
            if stage != "unknown":
                stage_votes[pid_of(global_tid)][stage] += 1
            else:
                pass
        elif (
            end is not None and stage == "runner" and annotation.get("op") == "execute"
        ):
            execute_ranges[pid_of(global_tid)].append(
                (max(start, t0), min(end, t1), str(annotation.get("mode", "?")))
            )
        else:
            pass
    stage_by_pid = {
        pid: votes.most_common(1)[0][0] for pid, votes in stage_votes.items()
    }

    events = db.execute(
        "select timestamp, tag, contextId, globalPid from GPU_CONTEXT_SWITCH_EVENTS order by timestamp, seqNo"
    ).fetchall()
    open_since: dict[int, int] = {}
    last_ack: dict[int, str] = {}
    pid_by_context: dict[int, int] = {}
    slices: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    switch_gaps: list[float] = []
    last_save: int | None = None
    for timestamp, tag, context, global_pid in events:
        pid_by_context[context] = pid_of(global_pid)
        if tag in ACK_TAGS:
            last_ack[context] = ACK_TAGS[tag]
        elif tag == RESTORE_START:
            open_since[context] = timestamp
            if last_save is not None and t0 <= timestamp <= t1:
                switch_gaps.append((timestamp - last_save) / 1e3)
            else:
                pass
        elif tag == SAVE_END and context in open_since:
            start = open_since.pop(context)
            if timestamp > t0 and start < t1:
                slices[context].append(
                    (max(start, t0), min(timestamp, t1), last_ack.pop(context, "none"))
                )
            else:
                pass
            last_save = timestamp
        else:
            pass

    kernels_by_pid: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for start, end, global_pid in db.execute(
        "select max(start, ?), min(end, ?), globalPid from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?",
        (t0, t1, t0, t1),
    ):
        kernels_by_pid[pid_of(global_pid)].append((start, end))
    kernel_union = {pid: merged(intervals) for pid, intervals in kernels_by_pid.items()}

    name = lambda pid: f"{stage_by_pid.get(pid, '?')} ({pid})"
    print(f"window {window_ns / 1e6:.1f} ms, context switch events {len(events)}")
    print("1. residency per context")
    residency_by_pid: dict[int, list[tuple[int, int]]] = {}
    for context, context_slices in sorted(
        slices.items(), key=lambda item: -sum(e - s for s, e, _ in item[1])
    ):
        pid = pid_by_context[context]
        spans = merged([(start, end) for start, end, _ in context_slices])
        residency_by_pid[pid] = spans
        resident = sum(end - start for start, end in spans)
        own = kernel_union.get(pid, [])
        own_starts = [start for start, _ in own]
        busy_inside = sum(
            overlap_ns(own, own_starts, start, end) for start, end in spans
        )
        endings = collections.Counter(ack for _, _, ack in context_slices)
        print(
            f"  {name(pid):28s} slices {len(context_slices):7d} resident {resident / 1e6:9.1f} ms "
            f"({resident / window_ns:5.1%}) own kernels inside {busy_inside / 1e6:9.1f} ms, resident idle "
            f"{(resident - busy_inside) / 1e6:8.1f} ms"
        )
        print(
            f"  {'':28s} slice ms {quantiles([(end - start) / 1e6 for start, end, _ in context_slices])}"
        )
        print(f"  {'':28s} slice endings {dict(endings)}")
    print("2. switch cost, one context saved to the next restored (us)")
    print(
        f"  {len(switch_gaps)} switches, {quantiles(switch_gaps)}, total {sum(switch_gaps) / 1e3:.1f} ms"
    )

    print(
        "3. runner/execute ranges: wall, own residency, others' residency, switches in, own kernels (ms means)"
    )
    restore_times: dict[int, list[int]] = collections.defaultdict(list)
    for context, context_slices in slices.items():
        restore_times[pid_by_context[context]].extend(
            start for start, _, _ in context_slices
        )
    for pid in restore_times:
        restore_times[pid].sort()
    for pid, ranges in sorted(execute_ranges.items(), key=lambda item: name(item[0])):
        own_res = residency_by_pid.get(pid, [])
        own_res_starts = [start for start, _ in own_res]
        others = merged(
            [
                span
                for other, spans in residency_by_pid.items()
                if other != pid
                for span in spans
            ]
        )
        others_starts = [start for start, _ in others]
        own_kernels = kernel_union.get(pid, [])
        own_kernel_starts = [start for start, _ in own_kernels]
        by_mode: dict[str, list[tuple[float, float, float, float, float]]] = (
            collections.defaultdict(list)
        )
        for start, end, mode in ranges:
            if end <= start:
                continue
            switches = bisect.bisect_left(restore_times[pid], end) - bisect.bisect_left(
                restore_times[pid], start
            )
            by_mode[mode].append(
                (
                    (end - start) / 1e6,
                    overlap_ns(own_res, own_res_starts, start, end) / 1e6,
                    overlap_ns(others, others_starts, start, end) / 1e6,
                    float(switches),
                    overlap_ns(own_kernels, own_kernel_starts, start, end) / 1e6,
                )
            )
        for mode, rows in sorted(by_mode.items()):
            means = [statistics.fmean(column) for column in zip(*rows)]
            print(
                f"  {name(pid):28s} {mode:8s} ranges {len(rows):6d} wall {means[0]:7.2f} own resident {means[1]:7.2f} "
                f"others resident {means[2]:7.2f} switches in {means[3]:6.2f} own kernels {means[4]:7.2f}"
            )
    all_resident = merged(
        [span for spans in residency_by_pid.values() for span in spans]
    )
    print(
        f"4. no context resident: {(window_ns - sum(e - s for s, e in all_resident)) / 1e6:.1f} ms of {window_ns / 1e6:.1f}"
    )


if __name__ == "__main__":
    main()
