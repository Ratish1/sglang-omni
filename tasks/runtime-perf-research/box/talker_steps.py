"""The talker scheduler thread's steps in a card 0 nsys trace, cut at each step's end-of-step event wait.

usage: python3 - TRACE.sqlite SERVE.LOG < talker_steps.py

A step spans from the end of one cudaEventSynchronize on the talker's scheduler thread to the end of the next. A step
with more than one graph launch or more than 40 kernel launches is a prefill (the breakable prefill graph, the eager
sampler, the predictor and the emit), else a decode. Per kind: wall, time inside syncs, host time outside syncs, the
talker's own device time and other processes' device time inside the step, launch counts; then prefill trains.
"""

import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict

STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=(\d+) .*pid=(\d+)")


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def covered(intervals, start, end):
    return sum(max(0, min(b, end) - max(a, start)) for a, b in intervals if b > start and a < end)


def quantiles(values):
    ordered = sorted(values)
    return (statistics.median(ordered), ordered[int(0.9 * (len(ordered) - 1))]) if ordered else (0, 0)


def main() -> None:
    trace, serve_log = sys.argv[1], sys.argv[2]
    stages = {}
    with open(serve_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := STARTUP.search(line):
                stages[int(match[3])] = (match[1], int(match[2]))
    talker, device = next((pid, gpu) for pid, (stage, gpu) in stages.items() if stage == "talker_ar")
    db = sqlite3.connect(trace)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    calls_by_thread = defaultdict(list)
    for start, end, name_id, tid in db.execute(
        "SELECT start, end, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE ((globalTid >> 24) & 16777215) = ?",
        (talker,),
    ):
        calls_by_thread[tid & 0xFFFFFF].append((start, end, strings.get(name_id, str(name_id))))
    thread = max(calls_by_thread, key=lambda tid: len(calls_by_thread[tid]))
    calls = sorted(calls_by_thread[thread])
    own, others = [], []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        graph_filter = " AND (graphId IS NULL OR graphId = 0)" if table.endswith("KERNEL") else ""
        for start, end, global_pid in db.execute(f"SELECT start, end, globalPid FROM {table} WHERE deviceId = {device}{graph_filter}"):
            (own if pid_of(global_pid) == talker else others).append((start, end))
    own, others = merged(own), merged(others)
    boundaries = [end for start, end, name in calls if name.startswith("cudaEventSynchronize")]
    steps = []
    for left, right in zip(boundaries, boundaries[1:]):
        inside = [call for call in calls if left < call[0] and call[1] <= right]
        names = Counter(name.split("_v")[0] for _, _, name in inside)
        graphs = names["cudaGraphLaunch"]
        kernels = names["cudaLaunchKernel"] + names["cudaLaunchKernelExC"] + names["cuLaunchKernelEx"]
        sync = sum(end - start for start, end, name in inside if "Synchronize" in name)
        last_sync = next(call for call in calls if call[1] == right)
        sync += last_sync[1] - last_sync[0]
        stream_syncs = names["cudaStreamSynchronize"]
        kind = "prefill" if graphs > 1 else ("decode" if graphs == 1 else "no graph")
        steps.append({"kind": kind, "wall": (right - left) / 1e6, "sync": sync / 1e6, "host": (right - left - sync) / 1e6,
                      "own": covered(own, left, right) / 1e6, "others": covered(others, left, right) / 1e6,
                      "graphs": graphs, "kernels": kernels, "stream_syncs": stream_syncs, "memcpy": names["cudaMemcpyAsync"]})
    shapes = Counter((step["graphs"], step["kernels"] // 20 * 20, step["stream_syncs"]) for step in steps)
    print("step shapes (graph launches, kernel launches rounded down to 20, stream syncs): count, wall p50 ms")
    for shape, count in shapes.most_common(14):
        walls = [step["wall"] for step in steps if (step["graphs"], step["kernels"] // 20 * 20, step["stream_syncs"]) == shape]
        print(f"  {shape}: {count}, {statistics.median(walls):.1f}")
    print(f"talker pid {talker}, scheduler thread {thread}: {len(steps)} steps")
    print("| kind | steps | wall p50 / p90 ms | in syncs | host outside syncs | talker device | other stages' device | graph / kernel launches | stream syncs | copies |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for kind in ("prefill", "decode", "no graph"):
        rows = [step for step in steps if step["kind"] == kind]
        if not rows:
            continue
        else:
            pass
        wall = quantiles([row["wall"] for row in rows])
        mean = lambda key: statistics.mean(row[key] for row in rows)
        print(f"| {kind} | {len(rows)} | {wall[0]:.1f} / {wall[1]:.1f} | {mean('sync'):.1f} | {mean('host'):.1f} | "
              f"{mean('own'):.1f} | {mean('others'):.1f} | {mean('graphs'):.1f} / {mean('kernels'):.0f} | "
              f"{mean('stream_syncs'):.1f} | {mean('memcpy'):.1f} |")
    trains, run = Counter(), 0
    for step in steps:
        if step["kind"] == "prefill":
            run += 1
        else:
            if run:
                trains[run] += 1
            else:
                pass
            run = 0
    print("prefill trains (consecutive prefill steps between decodes):", dict(sorted(trains.items())))


if __name__ == "__main__":
    main()
