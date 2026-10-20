"""Device and host time of every section of the Fun-CosyVoice3 pipeline per finished request, and the AR's prefill
steps taken apart, from one nsys report taken with cosy_profile_probe (graph level CUDA graph trace).

A. Device time by section: every kernel, copy, memset and graph replay in the window, attributed through its
   launching runtime call (correlation id) to the thread that made it and that thread's innermost probe range at the
   call: AR <step mode> / <range> on the engine thread (the mode of the loop.step or loop.launch around the call),
   vocoder <plan> / <range> on the vocoder thread (the plan of the voc.step around it, or between steps),
   preprocessing / <range> on threads with pre.* ranges, and other. Per section: device ms summed over its work (not
   a union), ms per finished request (a vocoder.out result put in the window), share of the window. Then each
   role's union and the card's.
B. The AR's steps by mode: count, wall, own device time (union of the work the engine launched inside the step),
   runtime launch calls, rows, and the engine thread's time inside the step by state (GIL wait, CUDA wait, innermost
   range).
C. What a prefill costs the running decode: for each EXTEND step, the time from the end of the decode step before it
   to the start of the decode step after it, against the median of that time between two decode steps with nothing
   between them; the excess is the prefill's stall of every running row. Then the engine thread's time by state in
   that gap before the EXTEND step and after it.

usage: python3 section_ledger_c20261020.py REPORT.sqlite [--skip-head-s S] [--skip-tail-s S] [--top N]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics

from cosy_ledger import RANGE_EVENTS, Intervals, Thread, has_table, pid_of, short

GIL_WAITS = ("pthread_cond_timedwait", "pthread_cond_clockwait")
SEM_WAITS = ("sem_wait", "sem_timedwait", "sem_clockwait")
BLOCKING_CALLS = {"cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize", "cudaMemcpy", "cudaMemcpy2D"}
PROBE_PREFIXES = ("loop.", "mr.", "voc.", "flow.", "hift.", "pre.", "sched.")
STEP = re.compile(r"loop\.(step|launch) (\S+)(?: bs=(\d+))?")
PLAN = re.compile(r"voc\.step (\S+)")
RESULT = re.compile(r"q vocoder\.out\.put rid=\S+ t=result")
MS = 1e6


def stats(values: list[float]) -> str:
    if not values:
        return "n 0"
    ordered = sorted(values)
    return (
        f"n {len(values)} mean {statistics.mean(values):.2f} p50 {ordered[len(ordered) // 2]:.2f} "
        f"p90 {ordered[int(0.9 * (len(ordered) - 1))]:.2f} max {ordered[-1]:.2f}"
    )


def enclosing(spans: list[tuple[int, int, str]], starts: list[int], t: int) -> str | None:
    index = bisect.bisect_right(starts, t) - 1
    if index >= 0 and spans[index][1] > t:
        return spans[index][2]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--skip-head-s", type=float, default=1.0)
    parser.add_argument("--skip-tail-s", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))

    work = []
    for table, kind in (
        ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
        ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "graph"),
    ):
        if has_table(db, table):
            work += [(*row, kind) for row in db.execute(f"select start, end, correlationId, globalPid from {table}")]
    t0 = min(w[0] for w in work) + int(args.skip_head_s * 1e9)
    t1 = max(w[1] for w in work) - int(args.skip_tail_s * 1e9)
    window = t1 - t0
    work = [w for w in work if w[1] > t0 and w[0] < t1]

    ranges_by_tid: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    finished = 0
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS"
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label and RESULT.match(label) and t0 <= start <= t1:
            finished += 1
        elif end is not None and label.startswith(PROBE_PREFIXES):
            ranges_by_tid[tid].append((start, end, label))
        else:
            pass
    engine = {t for t, r in ranges_by_tid.items() if any(STEP.match(x[2]) for x in r)}
    vocoder = {t for t, r in ranges_by_tid.items() if any(x[2].startswith("voc.step") for x in r)}
    pre = {t for t, r in ranges_by_tid.items() if any(x[2].startswith("pre.") for x in r)} - engine - vocoder

    outer: dict[int, tuple[list[tuple[int, int, str]], list[int]]] = {}
    for tid in engine | vocoder:
        spans = sorted(
            r for r in ranges_by_tid[tid] if STEP.match(r[2]) or r[2].startswith("voc.step")
        )
        outer[tid] = (spans, [s for s, _, _ in spans])

    launcher: dict[tuple[int, int], tuple[int, int]] = {}
    launches_by_tid: dict[int, list[int]] = collections.defaultdict(list)
    blocking_by_tid: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for start, end, tid, corr, name_id in db.execute(
        "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        launcher[(pid_of(tid), corr)] = (tid, start)
        name = strings.get(name_id, "").split("_v")[0]
        if "Launch" in name:
            launches_by_tid[tid].append(start)
        elif name in BLOCKING_CALLS:
            blocking_by_tid[tid].append((start, end))
        else:
            pass
    for tid in launches_by_tid:
        launches_by_tid[tid].sort()
    gil_by_tid: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    sem_by_tid: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    if has_table(db, "OSRT_API"):
        for start, end, tid, name_id in db.execute("select start, end, globalTid, nameId from OSRT_API"):
            name = strings.get(name_id, "")
            if name.startswith(GIL_WAITS):
                gil_by_tid[tid].append((start, end))
            elif name.startswith(SEM_WAITS):
                sem_by_tid[tid].append((start, end))
            else:
                pass
    threads = {
        tid: Thread(ranges_by_tid[tid], gil_by_tid[tid], sem_by_tid[tid], blocking_by_tid[tid])
        for tid in engine | vocoder | pre
    }

    sections: collections.Counter = collections.Counter()
    by_role: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    own_by_tid: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    for start, end, corr, gpid, kind in work:
        found = launcher.get((pid_of(gpid), corr))
        clipped = (max(start, t0), min(end, t1))
        if found is None:
            sections["unattributed"] += clipped[1] - clipped[0]
            by_role["other"].append(clipped)
            continue
        else:
            pass
        tid, call = found
        own_by_tid[tid].append((call, start, end))
        if tid in engine:
            step = enclosing(*outer[tid], call)
            match = STEP.match(step) if step else None
            mode = match[2] if match else "between steps"
            key, role = f"AR {mode} / {threads[tid].label_at(call)}", "AR"
        elif tid in vocoder:
            step = enclosing(*outer[tid], call)
            match = PLAN.match(step) if step else None
            plan = match[1] if match else "between steps"
            key, role = f"vocoder {plan} / {threads[tid].label_at(call)}", "vocoder"
        elif tid in pre:
            key, role = f"preprocessing / {threads[tid].label_at(call)}", "preprocessing"
        else:
            key, role = f"other pid {pid_of(tid)} tid {tid & 0xFFFFFF}", "other"
        sections[f"{key} [{kind}]"] += clipped[1] - clipped[0]
        by_role[role].append(clipped)

    per_request = max(finished, 1)
    card = Intervals([(max(s, t0), min(e, t1)) for s, e, *_ in work])
    card_busy = sum(e - s for s, e in card.merged)
    print(f"window {window / 1e9:.2f} s, finished requests {finished}, card busy {100 * card_busy / window:.1f} %")
    print(f"threads: engine {len(engine)}, vocoder {len(vocoder)}, preprocessing {len(pre)}")
    print("\nA. device time by section (summed): ms in the window, ms per finished request, share of the window")
    total = sum(sections.values())
    for key, t in sections.most_common(args.top):
        print(f"  {key[:78]:78s} {t / MS:9.1f} {t / MS / per_request:8.3f} {100 * t / window:6.2f} %")
    print(f"  {'all sections':78s} {total / MS:9.1f} {total / MS / per_request:8.3f} {100 * total / window:6.2f} %")
    print("  each role's union: ms, ms per finished request, share of the window")
    for role, intervals in sorted(by_role.items()):
        busy = sum(e - s for s, e in Intervals(intervals).merged)
        print(f"  {role:16s} {busy / MS:9.1f} {busy / MS / per_request:8.3f} {100 * busy / window:6.2f} %")

    for tid in sorted(engine):
        spans, _ = outer[tid]
        steps = [s for s in spans if STEP.match(s[2]) and t0 <= s[0] and s[1] <= t1]
        own = sorted(own_by_tid[tid])
        own_calls = [c for c, _, _ in own]
        launches = launches_by_tid[tid]
        print(f"\nB. engine thread {tid & 0xFFFFFF}: steps by mode")
        by_mode: dict[str, list[tuple[int, int, str]]] = collections.defaultdict(list)
        for step in steps:
            by_mode[STEP.match(step[2])[2]].append(step)
        for mode, mode_steps in sorted(by_mode.items()):
            walls, devices, calls, rows = [], [], [], []
            states: collections.Counter = collections.Counter()
            for start, end, label in mode_steps:
                walls.append((end - start) / MS)
                a, b = bisect.bisect_left(own_calls, start), bisect.bisect_right(own_calls, end)
                devices.append(sum(e - s for s, e in Intervals([(s, e) for _, s, e in own[a:b]]).merged) / MS)
                calls.append(bisect.bisect_right(launches, end) - bisect.bisect_left(launches, start))
                bs = STEP.match(label)[3]
                rows.append(int(bs) if bs else 0)
                states.update(threads[tid].states(start, end))
            count = len(mode_steps)
            print(f"  {mode}: {count} steps, {sum(walls):.1f} ms of wall ({100 * sum(walls) * MS / window:.1f} % of the window)")
            print(f"    wall ms {stats(walls)}")
            print(f"    own device ms {stats(devices)}")
            print(f"    rows {stats([float(r) for r in rows])}; launch calls per step {statistics.mean(calls):.0f}")
            print(
                "    thread per step (ms): "
                + ", ".join(f"{k} {v / MS / count:.2f}" for k, v in states.most_common(12))
            )

        others = Intervals(
            [(max(s, t0), min(e, t1)) for t, items in own_by_tid.items() if t != tid for _, s, e in items]
        )
        alone: dict[int, list[float]] = collections.defaultdict(list)
        shared: dict[int, list[float]] = collections.defaultdict(list)
        for start, end, label in steps:
            if STEP.match(label)[2] != "DECODE":
                continue
            else:
                pass
            a, b = bisect.bisect_left(own_calls, start), bisect.bisect_right(own_calls, end)
            merged = Intervals([(s, e) for _, s, e in own[a:b]]).merged
            device = sum(e - s for s, e in merged)
            overlap = sum(others.covered(s, e) for s, e in merged)
            bs = STEP.match(label)[3]
            bucket = (int(bs) if bs else 0) // 8 * 8
            if device > 0 and overlap <= 0.05 * device:
                alone[bucket].append(device / MS)
            elif device > 0:
                shared[bucket].append(device / MS)
            else:
                pass
        print("  DECODE own device ms by rows (bins of 8): with other threads' work under under 5 % of it, and the rest")
        for bucket in sorted(set(alone) | set(shared)):
            print(f"    rows {bucket}-{bucket + 7}: alone {stats(alone[bucket])}; shared {stats(shared[bucket])}")

        decode = [s for s in steps if STEP.match(s[2])[2] == "DECODE"]
        extend = [s for s in steps if STEP.match(s[2])[2] == "EXTEND"]
        decode_starts = [s for s, _, _ in decode]
        extend_starts = [s for s, _, _ in extend]
        plain_gaps = []
        for before, after in zip(decode, decode[1:]):
            between = bisect.bisect_left(extend_starts, after[0]) - bisect.bisect_right(extend_starts, before[1])
            if between == 0:
                plain_gaps.append((after[0] - before[1]) / MS)
            else:
                pass
        typical = statistics.median(plain_gaps) if plain_gaps else 0.0
        stalls, holes = [], []
        before: collections.Counter = collections.Counter()
        after: collections.Counter = collections.Counter()
        for start, end, _ in extend:
            index = bisect.bisect_left(decode_starts, start)
            if 0 < index < len(decode):
                hole = (decode[index][0] - decode[index - 1][1]) / MS
                holes.append(hole)
                stalls.append(hole - typical)
                before.update(threads[tid].states(decode[index - 1][1], start))
                after.update(threads[tid].states(end, decode[index][0]))
            else:
                pass
        print("\nC. what a prefill costs the running decode")
        print(f"  gap between two decode steps with nothing between: median {typical:.2f} ms ({len(plain_gaps)} gaps)")
        print(f"  gap across a prefill: {stats(holes)}")
        print(f"  the prefill's excess, every running row stalled: {stats(stalls)}")
        print(
            f"  prefills {len(extend)} for {finished} finished requests; excess summed {sum(stalls):.0f} ms "
            f"({100 * sum(stalls) * MS / window:.1f} % of the window)"
        )
        count = max(len(holes), 1)
        for name, states in (("before the prefill", before), ("after the prefill", after)):
            print(
                f"  the engine thread {name}, to the decode step (ms a prefill): "
                + ", ".join(f"{k} {v / MS / count:.2f}" for k, v in states.most_common(12))
            )


if __name__ == "__main__":
    main()
