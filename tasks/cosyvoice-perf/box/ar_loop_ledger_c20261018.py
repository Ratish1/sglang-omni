"""AR engine loop ledger from one nsys report taken with the loop probe (OMNI_LOOP_PROBE=1).

Per engine scheduler thread (a thread with loop.step or loop.launch ranges):
- each step's device work: the kernels, copies and memsets whose launch (runtime call, matched by
  correlation id) the thread made inside the step's range, as a union of intervals;
- for consecutive decode steps with no idle between them: the step period, the engine's own device
  gap (its last kernel of step N to its first of step N+1), what the thread was doing during that
  gap (innermost probe range, "outside" between ranges), how much of the gap other threads' kernels
  kept the card busy, and the thread's blocking CUDA calls (synchronizes, synchronous copies and
  device to pageable host copies);
- the bound the async decode loop puts on the step period: per step max(device time, host time
  outside blocking calls), against the measured period;
- 10-18 copy: the same per step means by the step's rows, in bins of --rows-bin, with the thread's
  time in the ranges --by-rows-labels, so two arms compare at equal rows.

usage: python ar_loop_ledger_c20261018.py REPORT.sqlite [--skip-head-s S] [--skip-tail-s S] [--top N]
       [--rows-bin B] [--by-rows-labels L,L]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import sqlite3
import statistics

RANGE_EVENTS = (59, 60)
BLOCKING_CALLS = {
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
    "cudaMemcpy",
    "cudaMemcpy2D",
}
STEP_LABELS = ("loop.step", "loop.launch")
# A wait counts as a dependency inside the step when more device work than this is launched
# after it within the step; below it, the wait is a read of the step's results.
DEPENDENT_WORK_NS = 20_000
# note: on CPython 3.12 a contended interpreter lock acquire is pthread_cond_timedwait, never
# called when the lock is free; Python locks and queues wait in sem_wait and sem_clockwait.
GIL_WAITS = ("pthread_cond_timedwait", "pthread_cond_clockwait")
SEM_WAITS = ("sem_wait", "sem_timedwait", "sem_clockwait")


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def has_table(db: sqlite3.Connection, name: str) -> bool:
    row = db.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


def union(intervals: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def covered(merged: list[list[int]], starts: list[int], a: int, b: int) -> int:
    total = 0
    index = max(bisect.bisect_right(starts, a) - 1, 0)
    while index < len(merged) and merged[index][0] < b:
        start, end = merged[index]
        total += max(0, min(end, b) - max(start, a))
        index += 1
    return total


def flatten(ranges: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Nested ranges of one thread as non overlapping segments of the innermost range."""
    segments: list[tuple[int, int, str]] = []
    stack: list[tuple[int, str]] = []
    cursor = None

    def emit(a, b, label):
        if a is not None and b > a:
            segments.append((a, b, label))

    for start, end, label in sorted(ranges, key=lambda r: (r[0], -r[1])):
        while stack and stack[-1][0] <= start:
            top_end, top = stack.pop()
            emit(cursor, top_end, top)
            cursor = top_end
        emit(cursor, start, stack[-1][1] if stack else "outside")
        stack.append((end, label))
        cursor = start
    while stack:
        top_end, top = stack.pop()
        emit(cursor, top_end, top)
        cursor = top_end
    return segments


def breakdown(segments, starts, a: int, b: int) -> collections.Counter:
    out: collections.Counter = collections.Counter()
    index = max(bisect.bisect_right(starts, a) - 1, 0)
    while index < len(segments) and segments[index][0] < b:
        start, end, label = segments[index]
        overlap = min(end, b) - max(start, a)
        if overlap > 0:
            out[label] += overlap
        index += 1
    return out


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--skip-head-s", type=float, default=0.0)
    parser.add_argument("--skip-tail-s", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--rows-bin", type=int, default=4)
    parser.add_argument(
        "--by-rows-labels",
        default="mr.host_tokens,mr.collect,mr.emit,mr.forward,mr.sample,loop.next_batch,loop.result",
    )
    args = parser.parse_args()
    by_rows_labels = args.by_rows_labels.split(",")
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))

    device_tables = [
        t
        for t in (
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_MEMSET",
            "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
        )
        if has_table(db, t)
    ]
    work = []
    for table in device_tables:
        work += db.execute(
            f"select start, end, correlationId, globalPid from {table}"
        ).fetchall()
    t0 = min(w[0] for w in work) + int(args.skip_head_s * 1e9)
    t1 = max(w[1] for w in work) - int(args.skip_tail_s * 1e9)
    window = t1 - t0

    thread_names = {
        tid: strings.get(name_id, "?")
        for name_id, tid in db.execute("select nameId, globalTid from ThreadNames")
    }
    ranges_by_tid: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS "
        f"where eventType in {RANGE_EVENTS} and end is not null"
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith(("loop.", "mr.")):
            ranges_by_tid[tid].append((start, end, label))
    engine_tids = [
        tid
        for tid, ranges in ranges_by_tid.items()
        if any(r[2].startswith(STEP_LABELS) for r in ranges)
    ]

    calls_by_tid: dict[int, list[tuple[int, int, int, str]]] = collections.defaultdict(
        list
    )
    for start, end, tid, corr, name_id in db.execute(
        "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        calls_by_tid[tid].append(
            (start, end, corr, strings.get(name_id, "").split("_v")[0])
        )

    # note: a device to pageable host copy holds its cudaMemcpyAsync call until the stream reaches
    # it and the copy ends (.tolist(), .cpu(), .item()), so the call is a wait on the GPU.
    pageable_reads = {
        (pid_of(gpid), corr)
        for corr, gpid in db.execute(
            "select correlationId, globalPid from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where copyKind = 2 and dstKind = 0"
        )
    }
    osrt_by_tid: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    sem_by_tid: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    if has_table(db, "OSRT_API"):
        for start, end, tid, name_id in db.execute(
            "select start, end, globalTid, nameId from OSRT_API"
        ):
            name = strings.get(name_id, "")
            if name.startswith(GIL_WAITS):
                osrt_by_tid[tid].append((start, end))
            elif name.startswith(SEM_WAITS):
                sem_by_tid[tid].append((start, end))

    launcher_of: dict[tuple[int, int], int] = {}
    for tid in engine_tids:
        for start, end, corr, name in calls_by_tid[tid]:
            launcher_of[(pid_of(tid), corr)] = tid

    all_busy = union([(max(s, t0), min(e, t1)) for s, e, _, _ in work if e > t0 and s < t1])
    print(
        f"window {window / 1e9:.2f} s, card busy {100 * sum(e - s for s, e in all_busy) / window:.1f} %"
    )

    for tid in engine_tids:
        ranges = ranges_by_tid[tid]
        steps = sorted(
            (s, e, label)
            for s, e, label in ranges
            if label.startswith(STEP_LABELS) and s >= t0 and e <= t1
        )
        idles = sorted(
            (s, e) for s, e, label in ranges if label.startswith("loop.idle")
        )
        idle_starts = [s for s, _ in idles]
        step_starts = [s for s, _, _ in steps]
        launches_of_step: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        own_work = []
        other_work = []
        for start, end, corr, gpid in work:
            if end <= t0 or start >= t1:
                continue
            if launcher_of.get((pid_of(gpid), corr)) == tid:
                own_work.append((start, end, corr, gpid))
            else:
                other_work.append((start, end))
        call_start = {
            corr: start for start, _, corr, _ in calls_by_tid[tid]
        }
        last_launch: dict[int, int] = {}
        work_by_launch: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        for start, end, corr, _ in own_work:
            launched = call_start.get(corr)
            if launched is None:
                continue
            index = bisect.bisect_right(step_starts, launched) - 1
            if index >= 0 and launched <= steps[index][1]:
                launches_of_step[index].append((start, end))
                last_launch[index] = max(last_launch.get(index, launched), launched)
                work_by_launch[index].append((launched, end - start))
        lock_waits = union(osrt_by_tid.get(tid, []))
        lock_starts = [s for s, _ in lock_waits]
        sem_waits = union(sem_by_tid.get(tid, []))
        sem_starts = [s for s, _ in sem_waits]
        blocking = sorted(
            (s, e)
            for s, e, corr, name in calls_by_tid[tid]
            if name in BLOCKING_CALLS or (pid_of(tid), corr) in pageable_reads
        )
        blocking_merged = union(blocking)
        blocking_starts = [s for s, _ in blocking_merged]
        others = union(other_work)
        other_starts = [s for s, _ in others]
        segments = flatten(ranges)
        segment_starts = [s for s, _, _ in segments]
        own_busy = union([(s, e) for s, e, _, _ in own_work])

        modes = collections.Counter(label.split(" ")[1] if " " in label else "?" for _, _, label in steps)
        print(
            f"\nengine thread {thread_names.get(tid, tid)} (pid {pid_of(tid)}): steps {len(steps)} "
            + ", ".join(f"{m} {n}" for m, n in modes.most_common())
            + f"; its device work {100 * sum(e - s for s, e in own_busy) / window:.1f} % of the window"
        )

        periods, devices, gaps, waits, hosts, bounds, sizes, filled = [], [], [], [], [], [], [], []
        locks = []
        sems = []
        mid_waits = []
        lookahead_bounds = []
        after_last_wait = []
        waits_per_step = []
        host_mix: collections.Counter = collections.Counter()
        gap_mix: collections.Counter = collections.Counter()
        lock_mix: collections.Counter = collections.Counter()
        wait_mix: collections.Counter = collections.Counter()
        step_mixes: list[collections.Counter] = []
        for index in range(len(steps) - 1):
            this, nxt = steps[index], steps[index + 1]
            if " DECODE " not in f"{this[2]} " or " DECODE " not in f"{nxt[2]} ":
                continue
            j = bisect.bisect_left(idle_starts, this[0])
            if j < len(idles) and idles[j][0] < nxt[0]:
                continue
            this_work = union(launches_of_step.get(index, []))
            next_work = union(launches_of_step.get(index + 1, []))
            if not this_work or not next_work:
                continue
            period = nxt[0] - this[0]
            device = sum(e - s for s, e in this_work)
            gap_a, gap_b = this_work[-1][1], next_work[0][0]
            gap = max(0, gap_b - gap_a)
            wait = covered(blocking_merged, blocking_starts, this[0], nxt[0])
            host = period - wait
            k = bisect.bisect_left(blocking_starts, this[0])
            step_waits = []
            while k < len(blocking_merged) and blocking_merged[k][0] < this[1]:
                step_waits.append(blocking_merged[k])
                k += 1
            waits_per_step.append(len(step_waits))
            mid_wait = 0
            for wait_start, wait_end in step_waits:
                work_after = sum(
                    d for launched, d in work_by_launch[index] if launched >= wait_end
                )
                if work_after > DEPENDENT_WORK_NS:
                    mid_wait += min(wait_end, nxt[0]) - wait_start
            if step_waits:
                last_wait_end = step_waits[-1][1]
                after_last_wait.append(
                    sum(d for launched, d in work_by_launch[index] if launched >= last_wait_end)
                )
            periods.append(period)
            devices.append(device)
            gaps.append(gap)
            waits.append(wait)
            hosts.append(host)
            mid_waits.append(mid_wait)
            bounds.append(max(device, host))
            lookahead_bounds.append(max(device, host + mid_wait))
            sizes.append(int(this[2].split("bs=")[1]) if "bs=" in this[2] else 0)
            step_mix = breakdown(segments, segment_starts, this[0], nxt[0])
            step_mixes.append(step_mix)
            host_mix.update(step_mix)
            locks.append(covered(lock_waits, lock_starts, this[0], nxt[0]))
            sems.append(covered(sem_waits, sem_starts, this[0], nxt[0]))
            k = max(bisect.bisect_right(blocking_starts, this[0]) - 1, 0)
            while k < len(blocking_merged) and blocking_merged[k][0] < nxt[0]:
                a = max(blocking_merged[k][0], this[0])
                b = min(blocking_merged[k][1], nxt[0])
                if b > a:
                    wait_mix.update(breakdown(segments, segment_starts, a, b))
                k += 1
            k = max(bisect.bisect_right(lock_starts, this[0]) - 1, 0)
            while k < len(lock_waits) and lock_waits[k][0] < nxt[0]:
                a, b = max(lock_waits[k][0], this[0]), min(lock_waits[k][1], nxt[0])
                if b > a:
                    lock_mix.update(breakdown(segments, segment_starts, a, b))
                k += 1
            if gap:
                gap_mix.update(breakdown(segments, segment_starts, gap_a, gap_b))
                filled.append(covered(others, other_starts, gap_a, gap_b) / gap)
        if not periods:
            print("  no consecutive decode steps in the window")
            continue
        n = len(periods)
        ms = 1e6
        print(
            f"  consecutive decode steps {n}, rows per step mean {statistics.mean(sizes):.1f}"
            f"\n  period mean {statistics.mean(periods) / ms:.2f} ms (p50 {percentile(periods, 0.5) / ms:.2f}, p90 {percentile(periods, 0.9) / ms:.2f})"
            f"\n  own device time per step {statistics.mean(devices) / ms:.2f} ms ({100 * sum(devices) / sum(periods):.1f} % of the period)"
            f"\n  own device gap per step {statistics.mean(gaps) / ms:.2f} ms ({100 * sum(gaps) / sum(periods):.1f} % of the period),"
            f" other threads' work covers {100 * statistics.mean(filled) if filled else 0:.1f} % of it"
            f"\n  own device idle inside the step {statistics.mean(p - d - g for p, d, g in zip(periods, devices, gaps)) / ms:.2f} ms"
            f"\n  blocking CUDA calls per step {statistics.mean(waits) / ms:.2f} ms, host time outside them {statistics.mean(hosts) / ms:.2f} ms"
            f"\n  GIL waits (pthread_cond_timedwait) per step {statistics.mean(locks) / ms:.2f} ms,"
            f" Python lock and queue waits (sem_*) {statistics.mean(sems) / ms:.2f} ms"
            f"\n  of the blocking, waits with more than {DEPENDENT_WORK_NS / 1e3:.0f} us of device work launched after them in the step"
            f" (dependencies inside the step) {statistics.mean(mid_waits) / ms:.2f} ms,"
            f" the rest (reads of the step's results) {statistics.mean(w - m for w, m in zip(waits, mid_waits)) / ms:.2f} ms"
            f"\n  blocking calls per step {statistics.mean(waits_per_step):.1f}; device time launched after the step's last wait"
            f" {statistics.mean(after_last_wait) / ms if after_last_wait else 0:.3f} ms per step with a wait"
            f"\n  lookahead bound: period max(device, host + waits inside the step) {statistics.mean(lookahead_bounds) / ms:.2f} ms"
            f" ({100 * (sum(lookahead_bounds) / sum(periods) - 1):+.1f} %)"
            f"\n  no host wait bound: period max(device, host) {statistics.mean(bounds) / ms:.2f} ms against {statistics.mean(periods) / ms:.2f} ms"
            f" ({100 * (sum(bounds) / sum(periods) - 1):+.1f} %)"
        )
        print("  thread time per step by innermost range (ms):")
        print(
            "    "
            + ", ".join(
                f"{label} {t / n / ms:.2f}" for label, t in host_mix.most_common(args.top)
            )
        )
        if wait_mix:
            print("  blocking CUDA calls per step by innermost range (ms):")
            print(
                "    "
                + ", ".join(
                    f"{label} {t / n / ms:.2f}"
                    for label, t in wait_mix.most_common(args.top)
                )
            )
        if lock_mix:
            print("  GIL waits per step by innermost range (ms):")
            print(
                "    "
                + ", ".join(
                    f"{label} {t / n / ms:.2f}"
                    for label, t in lock_mix.most_common(args.top)
                )
            )
        print("  thread time inside the own device gap per step (ms):")
        print(
            "    "
            + ", ".join(
                f"{label} {t / n / ms:.2f}" for label, t in gap_mix.most_common(args.top)
            )
        )
        print(
            f"  by rows (bins of {args.rows_bin}; ms per step): steps, period, device, blocking, host outside it,"
            " GIL waits, " + ", ".join(by_rows_labels)
        )
        bins: dict[int, list[int]] = collections.defaultdict(list)
        for k, size in enumerate(sizes):
            bins[(size - 1) // args.rows_bin].append(k)
        for key in sorted(bins):
            members = bins[key]
            if len(members) < 20:
                continue
            mean = lambda values: statistics.mean(values[k] for k in members) / ms
            label_means = [
                statistics.mean(
                    sum(t for label, t in step_mixes[k].items() if label.startswith(wanted))
                    for k in members
                )
                / ms
                for wanted in by_rows_labels
            ]
            print(
                f"    rows {key * args.rows_bin + 1}-{(key + 1) * args.rows_bin}: {len(members)},"
                f" {mean(periods):.2f}, {mean(devices):.2f}, {mean(waits):.2f}, {mean(hosts):.2f},"
                f" {mean(locks):.2f}, " + ", ".join(f"{value:.2f}" for value in label_means)
            )


if __name__ == "__main__":
    main()
