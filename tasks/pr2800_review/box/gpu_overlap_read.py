"""Reader for gil_wait_check.py traces: GPU time of the talker thread's and the code2wav thread's work, and how much
of the code2wav work runs while talker work is on the device.

Run on the box: python3 /data/pr2800/gpu_overlap_read.py /data/pr2800/<run>
Graph executions and kernels of the talker process are attributed to the thread that launched them (the runtime
call with the same correlation id); the two threads with the most launches are the talker and code2wav threads.
Per arm: each thread's executions, summed and merged GPU time, and the merged code2wav time that overlaps merged
talker time.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


def process_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def overlap(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> int:
    total, i, j = 0, 0, 0
    while i < len(first) and j < len(second):
        start, end = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        total += max(0, end - start)
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return total


def main() -> None:
    out = Path(sys.argv[1])
    for record in json.loads((out / "summary.json").read_text()):
        arm, talker_pid = record["arm"], record["pids"]["talker_ar"]
        connection = sqlite3.connect(out / arm / "trace.sqlite")
        thread_of: dict[int, int] = {}
        launches: dict[int, int] = defaultdict(int)
        for correlation, tid in connection.execute("SELECT correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
            if process_of(tid) == talker_pid:
                thread_of[correlation] = tid
                launches[tid] += 1
            else:
                pass
        intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for table in ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_KERNEL"):
            for start, end, correlation in connection.execute(f"SELECT start, end, correlationId FROM {table}"):
                tid = thread_of.get(correlation)
                if tid is not None:
                    intervals[tid].append((start, end))
                else:
                    pass
        talker, code2wav = sorted(launches, key=lambda tid: -launches[tid])[:2]
        merged_talker, merged_code2wav = merged(intervals[talker]), merged(intervals[code2wav])
        print(f"== {arm} (card {record['card']})")
        for name, tid, spans in (("talker", talker, merged_talker), ("code2wav", code2wav, merged_code2wav)):
            summed = sum(end - start for start, end in intervals[tid]) / 1e6
            busy = sum(end - start for start, end in spans) / 1e6
            print(f"  {name:8s} executions {len(intervals[tid]):7d} summed {summed:8.1f} ms merged {busy:8.1f} ms")
        print(f"  code2wav merged time overlapping talker work {overlap(merged_code2wav, merged_talker) / 1e6:.1f} ms")


if __name__ == "__main__":
    main()
