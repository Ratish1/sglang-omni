"""Device and host time per stage process and per pipeline NVTX op (run on the box).

Reads an nsys sqlite export of a serve captured with SGLANG_OMNI_PIPELINE_NVTX=1 (the
prof branch). Window: the two epoch seconds in --window (run_nsys_boot.sh writes the
generation start and end); without it, first to last kernel.

Prints:
  1. per process, named by the stage its recorder marks carry: kernels, busy time (union
     of its kernel intervals, overlap counted once), share of the window, memcpy time;
     and the union over all processes (the GPU busy share).
  2. per NVTX op ("stage/op" from the annotation): ranges, host ms total and mean, and
     the device ms of the kernels launched inside it. A kernel belongs to the innermost
     range that encloses the runtime call it correlates with, on the calling thread;
     graph node kernels correlate with their cudaGraphLaunch.

  3. per process, CUDA runtime calls with their host time.
  4. with --api-in-ranges STAGE: that process's runtime calls per innermost NVTX range
     (calls and host ms per range and call name), so syncs and copies land on the step
     part that issued them.

usage: python nsys_stage_ledger.py REPORT.sqlite [--window window.txt] [--top 40]
       [--api-in-ranges talker_ar]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import sqlite3

PREFIX = "omni.pipeline:"
RANGE_SCAN_LIMIT = 64


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def has_table(db: sqlite3.Connection, name: str) -> bool:
    row = db.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    current_start = current_end = None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def session_window(db: sqlite3.Connection, window_path: str | None) -> tuple[int, int]:
    if window_path is None:
        return db.execute(
            "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
    with open(window_path) as handle:
        begin_s, end_s = (float(line) for line in handle.read().split()[:2])
    session_ns = db.execute(
        "select utcEpochNs from TARGET_INFO_SESSION_START_TIME"
    ).fetchone()[0]
    return int(begin_s * 1e9) - session_ns, int(end_s * 1e9) - session_ns


def nvtx_rows(db: sqlite3.Connection, t0: int, t1: int):
    rows = db.execute(
        """
        select e.start, e.end, e.globalTid,
               coalesce(e.text, s.value) as text
        from NVTX_EVENTS e left join StringIds s on e.textId = s.id
        where e.start <= ? and coalesce(e.end, e.start) >= ?
        """,
        (t1, t0),
    )
    for start, end, global_tid, text in rows:
        if text is None or not text.startswith(PREFIX):
            continue
        yield start, end, global_tid, json.loads(text[len(PREFIX) :])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--api-in-ranges")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    t0, t1 = session_window(db, args.window)
    window_ms = (t1 - t0) / 1e6

    stage_votes: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    ranges_by_thread: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(
        list
    )
    for start, end, global_tid, annotation in nvtx_rows(db, t0, t1):
        stage = annotation.get("stage", "unknown")
        if annotation.get("kind") == "mark":
            if stage != "unknown":
                stage_votes[pid_of(global_tid)][stage] += 1
            continue
        if end is not None:
            ranges_by_thread[global_tid].append(
                (start, end, f"{stage}/{annotation.get('op')}")
            )
    stage_by_pid = {
        pid: votes.most_common(1)[0][0] for pid, votes in stage_votes.items()
    }
    # a range is named by the process that ran it, so runner/execute splits per stage
    for global_tid, ranges in ranges_by_thread.items():
        process = stage_by_pid.get(pid_of(global_tid), "?")
        ranges_by_thread[global_tid] = [
            (start, end, f"{process}:{op}") for start, end, op in ranges
        ]

    kernels = db.execute(
        """
        select max(start, ?), min(end, ?), globalPid, correlationId
        from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?
        """,
        (t0, t1, t0, t1),
    ).fetchall()
    intervals_by_pid: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    kernel_ns_by_correlation: dict[tuple[int, int], int] = collections.Counter()
    for start, end, global_pid, correlation in kernels:
        pid = pid_of(global_pid)
        intervals_by_pid[pid].append((start, end))
        kernel_ns_by_correlation[(pid, correlation)] += end - start

    memcpy_ns_by_pid: dict[int, int] = collections.Counter()
    if has_table(db, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for start, end, global_pid in db.execute(
            """
            select max(start, ?), min(end, ?), globalPid
            from CUPTI_ACTIVITY_KIND_MEMCPY where end > ? and start < ?
            """,
            (t0, t1, t0, t1),
        ):
            memcpy_ns_by_pid[pid_of(global_pid)] += end - start

    print(f"window {window_ms:.1f} ms")
    print("process                      kernels   busy_ms  share   memcpy_ms")
    all_intervals = []
    for pid, intervals in sorted(
        intervals_by_pid.items(), key=lambda item: -union_ns(item[1])
    ):
        busy = union_ns(intervals)
        all_intervals.extend(intervals)
        name = f"{stage_by_pid.get(pid, '?')} ({pid})"
        print(
            f"{name:28s} {len(intervals):8d} {busy / 1e6:9.1f} "
            f"{busy / 1e6 / window_ms:6.1%} {memcpy_ns_by_pid[pid] / 1e6:10.1f}"
        )
    gpu_busy = union_ns(all_intervals)
    print(
        f"{'all processes':28s} {len(all_intervals):8d} {gpu_busy / 1e6:9.1f} {gpu_busy / 1e6 / window_ms:6.1%}"
    )

    for ranges in ranges_by_thread.values():
        ranges.sort()
    starts_by_thread = {
        tid: [start for start, _, _ in ranges]
        for tid, ranges in ranges_by_thread.items()
    }

    def innermost_range(global_tid: int, call_start: int) -> str:
        ranges = ranges_by_thread.get(global_tid)
        if not ranges:
            return "(no range)"
        else:
            pass
        index = bisect.bisect_right(starts_by_thread[global_tid], call_start) - 1
        for candidate in range(index, max(index - RANGE_SCAN_LIMIT, -1), -1):
            start, end, op = ranges[candidate]
            if start <= call_start <= end:
                return op
            else:
                pass
        return "(no range)"

    device_ns_by_op: dict[str, int] = collections.Counter()
    for call_start, global_tid, correlation in db.execute(
        """
        select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME
        where start >= ? and start < ?
        """,
        (t0, t1),
    ):
        kernel_ns = kernel_ns_by_correlation.get((pid_of(global_tid), correlation))
        if not kernel_ns:
            continue
        device_ns_by_op[innermost_range(global_tid, call_start)] += kernel_ns

    host_ns_by_op: dict[str, list[int]] = collections.defaultdict(list)
    for ranges in ranges_by_thread.values():
        for start, end, op in ranges:
            host_ns_by_op[op].append(end - start)
    print()
    print(
        "process:op                                        ranges   host_ms  host_mean_us  device_ms"
    )
    ops = set(host_ns_by_op) | set(device_ns_by_op)
    for op in sorted(ops, key=lambda name: -device_ns_by_op[name])[: args.top]:
        durations = host_ns_by_op.get(op, [])
        host_ms = sum(durations) / 1e6
        mean_us = host_ms * 1e3 / len(durations) if durations else 0.0
        print(
            f"{op:48s} {len(durations):7d} {host_ms:9.1f} {mean_us:13.1f} "
            f"{device_ns_by_op[op] / 1e6:10.1f}"
        )

    # CUDA runtime calls per process: syncs, copies and launches with their host time
    api_rows = db.execute(
        """
        select r.globalTid, s.value, r.end - r.start from CUPTI_ACTIVITY_KIND_RUNTIME r
        join StringIds s on r.nameId = s.id where r.start >= ? and r.start < ?
        """,
        (t0, t1),
    ).fetchall()
    api_by_stage: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for global_tid, name, duration in api_rows:
        stage = stage_by_pid.get(pid_of(global_tid), "?")
        api_by_stage[(stage, name.split("_v")[0])].append(duration)
    print()
    print("process:cuda api                                 calls   host_ms  mean_us")
    ordered = sorted(api_by_stage.items(), key=lambda item: -sum(item[1]))
    for (stage, name), durations in ordered[: args.top]:
        total_ms = sum(durations) / 1e6
        print(
            f"{stage + ':' + name:48s} {len(durations):7d} {total_ms:9.1f} "
            f"{total_ms * 1e3 / len(durations):8.1f}"
        )

    if args.api_in_ranges is None:
        return
    else:
        pass
    api_by_range: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for call_start, global_tid, name, duration in db.execute(
        """
        select r.start, r.globalTid, s.value, r.end - r.start
        from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId = s.id
        where r.start >= ? and r.start < ?
        """,
        (t0, t1),
    ):
        if stage_by_pid.get(pid_of(global_tid)) != args.api_in_ranges:
            continue
        else:
            pass
        owner = innermost_range(global_tid, call_start)
        api_by_range[(owner, name.split("_v")[0])].append(duration)
    print()
    print(
        f"{args.api_in_ranges} cuda api per innermost range         calls   host_ms  mean_us"
    )
    ordered = sorted(api_by_range.items(), key=lambda item: -sum(item[1]))
    for (owner, name), durations in ordered[: args.top]:
        total_ms = sum(durations) / 1e6
        print(
            f"{owner + ' ' + name:56s} {len(durations):7d} {total_ms:9.1f} "
            f"{total_ms * 1e3 / len(durations):8.1f}"
        )


if __name__ == "__main__":
    main()
