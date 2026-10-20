"""Fun-CosyVoice3 bubble and vocoder step ledger, from one nsys report taken with the combined
probe (cosy_profile_probe: loop.* and mr.* on the AR engine, voc.*, flow.*, hift.*, pre.* on the
pipeline) and the OS runtime trace.

A. Card idle: every interval in the window with no kernel, copy, memset or graph replay of any
   thread (a graph level trace reports a replay as one interval, not its kernels), by length, and each interval of at least --min-gap-us attributed to what the AR engine thread
   and the vocoder thread were doing over it: waiting for work (sem_* waits), waiting for the GIL
   (pthread_cond_timedwait), in a blocking CUDA call, or host work in their innermost probe range.
B. Vocoder steps (voc.step ranges), per plan: wall time, the vocoder thread's own device work
   (launched inside the step, matched by correlation id), the card's busy share over the step,
   the device idle inside the step and what the thread did in it, blocking calls, GIL waits, and
   the thread's time by innermost range; then the thread's time between steps.

usage: python cosy_ledger.py REPORT.sqlite [--skip-head-s S] [--skip-tail-s S] [--min-gap-us U]
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
GIL_WAITS = ("pthread_cond_timedwait", "pthread_cond_clockwait")
SEM_WAITS = ("sem_wait", "sem_timedwait", "sem_clockwait")
PROBE_PREFIXES = ("loop.", "mr.", "voc.", "flow.", "hift.", "pre.", "cap ", "compiled")
GAP_BUCKETS_US = (5, 10, 50, 100, 500, 1000, 5000, 10000)


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def has_table(db: sqlite3.Connection, name: str) -> bool:
    row = db.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


def union(intervals) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


class Intervals:
    """Merged intervals with coverage queries."""

    def __init__(self, intervals) -> None:
        self.merged = union(intervals)
        self.starts = [s for s, _ in self.merged]

    def covered(self, a: int, b: int) -> int:
        total = 0
        index = max(bisect.bisect_right(self.starts, a) - 1, 0)
        while index < len(self.merged) and self.merged[index][0] < b:
            start, end = self.merged[index]
            total += max(0, min(end, b) - max(start, a))
            index += 1
        return total

    def pieces(self, a: int, b: int):
        index = max(bisect.bisect_right(self.starts, a) - 1, 0)
        while index < len(self.merged) and self.merged[index][0] < b:
            start, end = self.merged[index]
            if min(end, b) > max(start, a):
                yield max(start, a), min(end, b)
            index += 1


def flatten(ranges):
    """Nested ranges of one thread as non overlapping segments of the innermost range."""
    segments = []
    stack = []
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


class Thread:
    """One thread's ranges, waits and blocking calls."""

    def __init__(self, ranges, gil, sem, blocking) -> None:
        self.segments = flatten(ranges)
        self.segment_starts = [s for s, _, _ in self.segments]
        self.gil = Intervals(gil)
        self.sem = Intervals(sem)
        self.blocking = Intervals(blocking)

    def label_at(self, t: int) -> str:
        index = bisect.bisect_right(self.segment_starts, t) - 1
        if index >= 0 and self.segments[index][1] > t:
            return short(self.segments[index][2])
        return "outside"

    def labels(self, a: int, b: int) -> collections.Counter:
        out: collections.Counter = collections.Counter()
        index = max(bisect.bisect_right(self.segment_starts, a) - 1, 0)
        while index < len(self.segments) and self.segments[index][0] < b:
            start, end, label = self.segments[index]
            overlap = min(end, b) - max(start, a)
            if overlap > 0:
                out[short(label)] += overlap
            index += 1
        return out

    def states(self, a: int, b: int) -> collections.Counter:
        """Time over [a, b) as queue wait, GIL wait, blocking call, or host work by label."""
        out: collections.Counter = collections.Counter()
        claimed = []
        for name, intervals in (("queue wait", self.sem), ("GIL wait", self.gil), ("CUDA wait", self.blocking)):
            for start, end in intervals.pieces(a, b):
                out[name] += end - start
                claimed.append((start, end))
        claimed = Intervals(claimed)
        cursor = a
        for start, end in claimed.merged + [[b, b]]:
            if start > cursor:
                for label, t in self.labels(cursor, min(start, b)).items():
                    out[label] += t
            cursor = max(cursor, end)
            if cursor >= b:
                break
        return out


def short(label: str) -> str:
    """A label without its per call fields (rid, rows, frames), kept to its kind."""
    words = label.split(" ")
    kept = [w for w in words if "=" not in w or w.startswith(("plan=", "streaming=", "final="))]
    return " ".join(kept)[:48]


def bucket_name(us: float) -> str:
    for edge in GAP_BUCKETS_US:
        if us < edge:
            return f"<{edge} us"
    return f">={GAP_BUCKETS_US[-1]} us"


def fmt(counter: collections.Counter, scale: float, top: int) -> str:
    total = sum(counter.values()) or 1
    return ", ".join(
        f"{label} {t / scale:.2f} ({100 * t / total:.0f} %)" for label, t in counter.most_common(top)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--skip-head-s", type=float, default=0.0)
    parser.add_argument("--skip-tail-s", type=float, default=0.0)
    parser.add_argument("--min-gap-us", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=12)
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
            work += [
                (*row, kind) for row in db.execute(f"select start, end, correlationId, globalPid from {table}")
            ]
    t0 = min(w[0] for w in work) + int(args.skip_head_s * 1e9)
    t1 = max(w[1] for w in work) - int(args.skip_tail_s * 1e9)
    window = t1 - t0
    work = [w for w in work if w[1] > t0 and w[0] < t1]

    ranges_by_tid = collections.defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS "
        f"where eventType in {RANGE_EVENTS} and end is not null"
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label.startswith(PROBE_PREFIXES):
            ranges_by_tid[tid].append((start, end, label))
    engine = [t for t, r in ranges_by_tid.items() if any(x[2].startswith("loop.step") for x in r)]
    vocoder = [t for t, r in ranges_by_tid.items() if any(x[2].startswith("voc.step") for x in r)]

    launcher = {}
    call_start = {}
    blocking_by_tid = collections.defaultdict(list)
    calls_by_tid = collections.defaultdict(list)
    pageable = {
        (pid_of(g), c)
        for c, g in db.execute(
            "select correlationId, globalPid from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 and dstKind = 0"
        )
    }
    for start, end, tid, corr, name_id in db.execute(
        "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        launcher[(pid_of(tid), corr)] = tid
        call_start[(pid_of(tid), corr)] = start
        name = strings.get(name_id, "").split("_v")[0]
        if (pid_of(tid), corr) in pageable:
            name += " pageable"
        calls_by_tid[tid].append((start, end, name))
        if name in BLOCKING_CALLS or name.endswith(" pageable"):
            blocking_by_tid[tid].append((start, end))
    gil_by_tid = collections.defaultdict(list)
    sem_by_tid = collections.defaultdict(list)
    if has_table(db, "OSRT_API"):
        for start, end, tid, name_id in db.execute("select start, end, globalTid, nameId from OSRT_API"):
            name = strings.get(name_id, "")
            if name.startswith(GIL_WAITS):
                gil_by_tid[tid].append((start, end))
            elif name.startswith(SEM_WAITS):
                sem_by_tid[tid].append((start, end))
    threads = {
        tid: Thread(ranges_by_tid[tid], gil_by_tid[tid], sem_by_tid[tid], blocking_by_tid[tid])
        for tid in set(engine) | set(vocoder)
    }

    busy = Intervals([(max(s, t0), min(e, t1)) for s, e, *_ in work])
    idle = []
    cursor = t0
    for start, end in busy.merged:
        if start > cursor:
            idle.append((cursor, start))
        cursor = max(cursor, end)
    if t1 > cursor:
        idle.append((cursor, t1))
    idle_total = sum(b - a for a, b in idle)
    ms = 1e6
    print(f"window {window / 1e9:.2f} s, card busy {100 * (window - idle_total) / window:.1f} %, idle {idle_total / ms:.1f} ms")
    print(f"engine threads {len(engine)}, vocoder threads {len(vocoder)}")

    print("\nA. card idle intervals by length: count, total ms, share of idle")
    buckets = collections.Counter()
    counts = collections.Counter()
    for a, b in idle:
        name = bucket_name((b - a) / 1e3)
        buckets[name] += b - a
        counts[name] += 1
    for edge in [f"<{e} us" for e in GAP_BUCKETS_US] + [f">={GAP_BUCKETS_US[-1]} us"]:
        if counts[edge]:
            print(f"  {edge:>10s}: {counts[edge]:8d}  {buckets[edge] / ms:9.1f} ms  {100 * buckets[edge] / idle_total:5.1f} %")

    min_gap = args.min_gap_us * 1e3
    long_idle = [(a, b) for a, b in idle if b - a >= min_gap]
    long_total = sum(b - a for a, b in long_idle)
    print(f"\n  idle intervals of at least {args.min_gap_us:.0f} us: {len(long_idle)}, {long_total / ms:.1f} ms")
    for role, tids in (("engine", engine), ("vocoder", vocoder)):
        for tid in tids:
            states = collections.Counter()
            for a, b in long_idle:
                states.update(threads[tid].states(a, b))
            print(f"  {role} thread over them (ms, share): {fmt(states, ms, args.top)}")
    if engine and vocoder:
        pairs = collections.Counter()
        for a, b in long_idle:
            e = threads[engine[0]].states(a, b).most_common(1)
            v = threads[vocoder[0]].states(a, b).most_common(1)
            pairs[(e[0][0] if e else "-", v[0][0] if v else "-")] += b - a
        print("  dominant (engine, vocoder) state per interval (ms):")
        for (e, v), t in pairs.most_common(args.top):
            print(f"    {t / ms:9.1f}  {100 * t / long_total:5.1f} %  engine: {e} | vocoder: {v}")

    for tid in vocoder:
        thread = threads[tid]
        steps = sorted(r for r in ranges_by_tid[tid] if r[2].startswith("voc.step") and r[0] >= t0 and r[1] <= t1)
        step_starts = [s for s, _, _ in steps]
        own = collections.defaultdict(list)
        own_by_range = collections.defaultdict(collections.Counter)
        for start, end, corr, gpid, kind in work:
            key = (pid_of(gpid), corr)
            if launcher.get(key) != tid:
                continue
            launched = call_start[key]
            index = bisect.bisect_right(step_starts, launched) - 1
            if index >= 0 and launched <= steps[index][1]:
                own[index].append((start, end))
                own_by_range[index][f"{thread.label_at(launched)} {kind}"] += end - start
        calls = sorted(calls_by_tid[tid])
        call_starts = [c[0] for c in calls]
        print(f"\nB. vocoder steps: {len(steps)} in the window")
        by_plan = collections.defaultdict(list)
        for index, (start, end, label) in enumerate(steps):
            plan = label.split(" ")[1] if " " in label else "?"
            rows = int(label.split("rows=")[1]) if "rows=" in label else 0
            mine = Intervals(own.get(index, []))
            device = mine.covered(start, end)
            tail = sum(e - max(s, end) for s, e in mine.merged if e > end)
            card = busy.covered(start, end)
            gaps = []
            cursor = start
            for s, e in mine.merged:
                if s > cursor and s < end:
                    gaps.append((cursor, min(s, end)))
                cursor = max(cursor, e)
            if cursor < end:
                gaps.append((cursor, end))
            gap_states = collections.Counter()
            for a, b in gaps:
                gap_states.update(thread.states(a, b))
            api = collections.Counter()
            api_time = collections.Counter()
            waits = collections.Counter()
            blocking_calls = collections.Counter()
            blocking_time = collections.Counter()
            for k in range(bisect.bisect_left(call_starts, start), bisect.bisect_left(call_starts, end)):
                c_start, c_end, c_name = calls[k]
                label = thread.label_at(c_start)
                api[(label, c_name)] += 1
                api_time[label] += c_end - c_start
                if c_name in BLOCKING_CALLS or c_name.endswith(" pageable"):
                    blocking_calls[(label, c_name)] += 1
                    blocking_time[(label, c_name)] += c_end - c_start
            for a, b in thread.blocking.pieces(start, end):
                waits[thread.label_at(a)] += b - a
            gil_labels = collections.Counter()
            for a, b in thread.gil.pieces(start, end):
                gil_labels[thread.label_at(a)] += b - a
            by_plan[plan].append(
                dict(
                    wall=end - start,
                    rows=rows,
                    device=device,
                    tail=tail,
                    card=card,
                    gap=sum(b - a for a, b in gaps),
                    gap_states=gap_states,
                    wait=thread.blocking.covered(start, end),
                    gil=thread.gil.covered(start, end),
                    labels=thread.labels(start, end),
                    api=api,
                    api_time=api_time,
                    waits=waits,
                    gil_labels=gil_labels,
                    own_by_range=own_by_range[index],
                    blocking_calls=blocking_calls,
                    blocking_time=blocking_time,
                )
            )
        for plan, rows in sorted(by_plan.items(), key=lambda kv: -sum(r["wall"] for r in kv[1])):
            n = len(rows)
            wall = sum(r["wall"] for r in rows)
            gap_states = collections.Counter()
            labels = collections.Counter()
            api = collections.Counter()
            api_time = collections.Counter()
            waits = collections.Counter()
            gil_labels = collections.Counter()
            own_by_range = collections.Counter()
            blocking_calls = collections.Counter()
            blocking_time = collections.Counter()
            for r in rows:
                gap_states.update(r["gap_states"])
                labels.update(r["labels"])
                api.update(r["api"])
                api_time.update(r["api_time"])
                waits.update(r["waits"])
                gil_labels.update(r["gil_labels"])
                own_by_range.update(r["own_by_range"])
                blocking_calls.update(r["blocking_calls"])
                blocking_time.update(r["blocking_time"])
            print(
                f"  plan {plan}: {n} steps, {wall / ms:.0f} ms ({100 * wall / window:.1f} % of the window),"
                f" rows mean {statistics.mean(r['rows'] for r in rows):.1f}"
                f"\n    wall per step mean {wall / n / ms:.2f} ms (p50 {statistics.median(r['wall'] for r in rows) / ms:.2f})"
                f", own device inside the step {sum(r['device'] for r in rows) / n / ms:.2f} ms"
                f" ({100 * sum(r['device'] for r in rows) / wall:.1f} %), own device after the step ends {sum(r['tail'] for r in rows) / n / ms:.2f} ms"
                f"\n    card busy over the step {100 * sum(r['card'] for r in rows) / wall:.1f} %,"
                f" own device idle inside the step {sum(r['gap'] for r in rows) / n / ms:.2f} ms"
                f"\n    blocking CUDA calls {sum(r['wait'] for r in rows) / n / ms:.2f} ms, GIL waits {sum(r['gil'] for r in rows) / n / ms:.2f} ms per step"
                f"\n    thread over its device idle inside the step (ms per step): {fmt(collections.Counter({k: v / n for k, v in gap_states.items()}), ms, args.top)}"
                f"\n    thread time per step by innermost range (ms): {fmt(collections.Counter({k: v / n for k, v in labels.items()}), ms, args.top)}"
                f"\n    blocking CUDA calls per step by innermost range (ms): {fmt(collections.Counter({k: v / n for k, v in waits.items()}), ms, args.top)}"
                f"\n    GIL waits per step by innermost range (ms): {fmt(collections.Counter({k: v / n for k, v in gil_labels.items()}), ms, args.top)}"
                f"\n    CUDA API calls per step {sum(api.values()) / n:.0f}, their time {sum(api_time.values()) / n / ms:.2f} ms;"
                f" by range: {fmt(collections.Counter({k: v / n for k, v in api_time.items()}), ms, args.top)}"
                f"\n    top CUDA API calls per step (range, call, count): "
                + ", ".join(f"{label} {name} {count / n:.0f}" for (label, name), count in api.most_common(args.top))
                + "\n    own device time per step by launching range and kind, summed (ms): "
                + fmt(collections.Counter({k: v / n for k, v in own_by_range.items()}), ms, args.top)
                + "\n    blocking CUDA calls per step (range, call, count, ms): "
                + ", ".join(
                    f"{label} {name} {blocking_calls[(label, name)] / n:.1f} {t / n / ms:.2f}"
                    for (label, name), t in blocking_time.most_common(args.top)
                )
            )
        between = collections.Counter()
        for (_, end, _), (start, _, _) in zip(steps, steps[1:]):
            between.update(thread.states(end, start))
        if steps:
            total = sum(between.values())
            print(
                f"  between steps: {total / ms:.0f} ms ({100 * total / window:.1f} % of the window):"
                f" {fmt(between, ms, args.top)}"
            )


if __name__ == "__main__":
    main()
