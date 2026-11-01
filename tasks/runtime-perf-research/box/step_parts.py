"""The talker's own device time inside its prefill and decode steps, by part: graph replays (each graph's span),
eager kernels by name, and copies, per step on average. Steps are cut as in talker_steps.py: from one end-of-step
cudaEventSynchronize on the talker's scheduler thread to the next; a prefill has more than one graph launch.

usage: python3 step_parts.py TRACE.sqlite SERVE.LOG
"""

import re
import sqlite3
import sys
from collections import Counter, defaultdict

STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=\d+ .*pid=(\d+)")


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


trace, serve_log = sys.argv[1], sys.argv[2]
stages = {}
with open(serve_log, encoding="utf-8", errors="replace") as log:
    for line in log:
        if match := STARTUP.search(line):
            stages[int(match[2])] = match[1]
talker = next(pid for pid, stage in stages.items() if stage == "talker_ar")
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
parts = []
for start, end, global_pid, graph_id in db.execute(
    "SELECT start, end, globalPid, graphId FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE deviceId = 0"
):
    if pid_of(global_pid) == talker:
        parts.append((start, end, f"graph {graph_id}"))
for start, end, global_pid, name_id in db.execute(
    "SELECT start, end, globalPid, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0 AND (graphId IS NULL OR graphId = 0)"
):
    if pid_of(global_pid) == talker:
        parts.append((start, end, "kernel " + strings.get(name_id, str(name_id))[:60]))
for start, end, global_pid in db.execute(
    "SELECT start, end, globalPid FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId = 0"
):
    if pid_of(global_pid) == talker:
        parts.append((start, end, "copy"))
parts.sort()
boundaries = [end for start, end, name in calls if name.startswith("cudaEventSynchronize")]
by_kind = {"prefill": Counter(), "decode": Counter()}
counts = Counter()
index = 0
for left, right in zip(boundaries, boundaries[1:]):
    graphs = sum(1 for call in calls if left < call[0] and call[1] <= right and call[2].startswith("cudaGraphLaunch"))
    kind = "prefill" if graphs > 1 else ("decode" if graphs == 1 else None)
    if kind is None:
        continue
    counts[kind] += 1
    for start, end, name in parts:
        if start >= right:
            break
        if end > left:
            label = "graph replays" if name.startswith("graph") else name
            by_kind[kind][label] += (min(end, right) - max(start, left)) / 1e6
for kind, totals in by_kind.items():
    steps = max(counts[kind], 1)
    print(f"{kind}: {counts[kind]} steps, own device time per step {sum(totals.values()) / steps:.2f} ms")
    for label, total in totals.most_common(14):
        print(f"  {total / steps:7.3f} ms  {label}")
