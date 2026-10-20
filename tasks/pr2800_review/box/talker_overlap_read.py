"""Reader for gil_wait_check.py traces: talker decode steps with and without code2wav work on the device.

Run on the box: python3 /data/pr2800/talker_overlap_read.py /data/pr2800/<run>
A talker step is the interval between two event synchronize ends on the talker thread (the thread of the talker
process with the most event synchronizes); a decode step is one under 20 ms. GPU executions are attributed to the
thread that launched them; code2wav's thread is the other thread with the most launches.
Per arm, for decode steps that overlap a code2wav execution and for those that do not: the count, the mean and
median step, and the talker's merged GPU time inside the step. Equal steps without code2wav and slower steps with it
put a gap on code2wav; slower steps in both put it elsewhere.
"""

from __future__ import annotations

import bisect
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from gpu_overlap_read import merged, process_of

DECODE_STEP_NS = 20_000_000


def busy_within(spans: list[tuple[int, int]], starts: list[int], start: int, end: int) -> int:
    total = 0
    index = max(0, bisect.bisect_right(starts, start) - 1)
    while index < len(spans) and spans[index][0] < end:
        total += max(0, min(end, spans[index][1]) - max(start, spans[index][0]))
        index += 1
    return total


def main() -> None:
    out = Path(sys.argv[1])
    for record in json.loads((out / "summary.json").read_text()):
        arm, talker_pid = record["arm"], record["pids"]["talker_ar"]
        connection = sqlite3.connect(out / arm / "trace.sqlite")
        names = dict(connection.execute("SELECT id, value FROM StringIds"))
        thread_of: dict[int, int] = {}
        launches: dict[int, int] = defaultdict(int)
        syncs: dict[int, list[int]] = defaultdict(list)
        for correlation, tid, end, name_id in connection.execute(
            "SELECT correlationId, globalTid, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ):
            if process_of(tid) != talker_pid:
                continue
            else:
                pass
            thread_of[correlation] = tid
            launches[tid] += 1
            if names.get(name_id, "").startswith("cudaEventSynchronize"):
                syncs[tid].append(end)
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
        talker = max(syncs, key=lambda tid: len(syncs[tid]))
        code2wav = max((tid for tid in launches if tid != talker), key=lambda tid: launches[tid])
        talker_spans, code2wav_spans = merged(intervals[talker]), merged(intervals[code2wav])
        talker_starts = [start for start, _ in talker_spans]
        code2wav_starts = [start for start, _ in code2wav_spans]
        ends = sorted(syncs[talker])
        groups: dict[str, list[tuple[float, float]]] = {"with code2wav": [], "without code2wav": []}
        for start, end in zip(ends, ends[1:]):
            if end - start >= DECODE_STEP_NS:
                continue
            else:
                pass
            shared = busy_within(code2wav_spans, code2wav_starts, start, end)
            own = busy_within(talker_spans, talker_starts, start, end)
            groups["with code2wav" if shared else "without code2wav"].append(((end - start) / 1e6, own / 1e6))
        print(f"== {arm} (card {record['card']})")
        for label, steps in groups.items():
            durations = [duration for duration, _ in steps]
            gpu = [own for _, own in steps]
            print(
                f"  {label:17s} steps {len(steps):5d} mean {statistics.mean(durations):6.3f} median "
                f"{statistics.median(durations):6.3f} ms, talker GPU mean {statistics.mean(gpu):6.3f} ms"
            )


if __name__ == "__main__":
    main()
