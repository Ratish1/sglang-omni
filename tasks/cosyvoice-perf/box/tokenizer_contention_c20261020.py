"""How a profile's speech tokenizer calls slow the AR and the vocoder in the same process.

For each label=sqlite (a cosy_profile_run c16 trace): AR decode periods and vocoder step walls split by overlap with a
pre.tokenize range, the AR thread's CUDA call time per step in each split, the tokenizer threads' copies by kind and
memory, and how many of the other threads' launch, copy and record calls over 1 ms end within 50 us of a tokenizer
thread's call that began before them.
usage: python3 tokenizer_contention_c20261020.py l2=<c16.sqlite> l3=<c16.sqlite>
"""

import bisect
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict

MEMCPY_KINDS = {1: "host to device", 2: "device to host", 8: "device to device"}
MEMORY_KINDS = {0: "pageable", 1: "pinned", 2: "device"}


def nvtx(db, where, parameters=()):
    return db.execute(
        "select n.start, n.end, coalesce(n.text, s.value), n.globalTid from NVTX_EVENTS n "
        f"left join StringIds s on s.id = n.textId where n.end is not null and {where} order by n.start",
        parameters,
    ).fetchall()


def overlaps(span_start, span_end, intervals, starts):
    i = max(0, bisect.bisect_right(starts, span_start) - 1)
    while i < len(intervals) and intervals[i][0] < span_end:
        if min(span_end, intervals[i][1]) > max(span_start, intervals[i][0]):
            return True
        else:
            i += 1
    return False


def summary(values):
    values = sorted(values)
    if values:
        return f"n {len(values)} mean {statistics.mean(values):.2f} p50 {values[len(values) // 2]:.2f}"
    else:
        return "n 0"


for argument in sys.argv[1:]:
    label, path = argument.split("=", 1)
    db = sqlite3.connect(path)
    calls = nvtx(db, "coalesce(n.text, s.value) = 'pre.tokenize'")
    tokenize = [(start, end) for start, end, _, _ in calls]
    starts = [start for start, _ in tokenize]
    tokenizer_threads = {thread for _, _, _, thread in calls}
    print(f"== {label} {path}: tokenizer calls {len(calls)}")

    steps = nvtx(db, "coalesce(n.text, s.value) like 'loop.step DECODE%'")
    engine_thread = Counter(thread for _, _, _, thread in steps).most_common(1)[0][0]
    steps = [start for start, _, _, thread in steps if thread == engine_thread]
    api = db.execute(
        "select r.start, r.end, s.value, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join StringIds s on s.id = r.nameId order by r.start"
    ).fetchall()
    engine_api = [row for row in api if row[3] == engine_thread]
    engine_api_starts = [row[0] for row in engine_api]
    periods = {"overlapped": [], "clear": []}
    api_ms = {"overlapped": defaultdict(float), "clear": defaultdict(float)}
    for step_start, next_start in zip(steps, steps[1:]):
        if next_start - step_start > 20e6:
            continue
        else:
            pass
        split = "overlapped" if overlaps(step_start, next_start, tokenize, starts) else "clear"
        periods[split].append((next_start - step_start) / 1e6)
        i = bisect.bisect_left(engine_api_starts, step_start)
        while i < len(engine_api) and engine_api[i][0] < next_start:
            api_ms[split][engine_api[i][2].split("_v")[0]] += (min(next_start, engine_api[i][1]) - engine_api[i][0]) / 1e6
            i += 1
    for split, values in periods.items():
        top = sorted(api_ms[split].items(), key=lambda pair: -pair[1])[:4]
        calls_per_step = ", ".join(f"{name} {total / max(1, len(values)):.2f}" for name, total in top)
        print(f"  AR decode period {split:10s} {summary(values)}; CUDA call ms per step: {calls_per_step}")

    for step_label in ("voc.step", "hift.step", "flow.prefix_graph"):
        walls = {"overlapped": [], "clear": []}
        for start, end, name, _ in nvtx(db, "coalesce(n.text, s.value) like ?", (step_label + "%",)):
            if name.split(" ")[0] == step_label:
                split = "overlapped" if overlaps(start, end, tokenize, starts) else "clear"
                walls[split].append((end - start) / 1e6)
            else:
                pass
        print(f"  {step_label} wall: " + "; ".join(f"{split} {summary(values)}" for split, values in walls.items()))

    copies = db.execute(
        "select m.copyKind, m.srcKind, m.dstKind, r.end - r.start, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r "
        "join CUPTI_ACTIVITY_KIND_MEMCPY m on m.correlationId = r.correlationId and m.globalPid / 16777216 = r.globalTid / 16777216"
    ).fetchall()
    by_kind = defaultdict(list)
    for copy_kind, source_kind, destination_kind, duration, thread in copies:
        if thread in tokenizer_threads:
            by_kind[(copy_kind, source_kind, destination_kind)].append(duration / 1e6)
        else:
            pass
    for (copy_kind, source_kind, destination_kind), durations in sorted(by_kind.items(), key=lambda pair: -sum(pair[1])):
        name = f"{MEMCPY_KINDS.get(copy_kind, copy_kind)} {MEMORY_KINDS.get(source_kind, source_kind)} to {MEMORY_KINDS.get(destination_kind, destination_kind)}"
        print(f"  tokenizer threads' copies {name}: n {len(durations)} call ms mean {statistics.mean(durations):.2f} max {max(durations):.1f} total {sum(durations):.0f}")

    tokenizer_ends = sorted((end, start) for start, end, _, thread in api if thread in tokenizer_threads)
    end_keys = [end for end, _ in tokenizer_ends]
    long_calls = [
        row
        for row in api
        if row[3] not in tokenizer_threads
        and row[1] - row[0] > 1e6
        and row[2].startswith(("cudaLaunchKernel", "cudaMemcpyAsync", "cudaEventRecord"))
    ]
    released = 0
    for start, end, _, _ in long_calls:
        i = bisect.bisect_left(end_keys, end - 50_000)
        while i < len(tokenizer_ends) and tokenizer_ends[i][0] <= end + 50_000:
            if tokenizer_ends[i][1] < start:
                released += 1
                break
            else:
                i += 1
    print(f"  other threads' launch, copy and record calls over 1 ms: {len(long_calls)}, ending with a tokenizer thread's call: {released}")
