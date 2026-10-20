"""Reader for gil_wait_check.py: per thread of the talker process, the waits the OS runtime trace saw.

Run on the box: python3 /data/pr2800/gil_read.py /data/pr2800/<run>
For each arm: the talker process's threads (named), their CUDA launches and event synchronizes, and the time they
spent in each pthread wait; the scheduler thread's waits per step (one event synchronize per talker step).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

WAITS = ("pthread_cond_timedwait", "pthread_cond_wait", "pthread_mutex_lock", "sem_wait", "sem_timedwait")


def process_of(global_tid: int) -> int:
    return (global_tid >> 24) & 0xFFFFFF


def main() -> None:
    out = Path(sys.argv[1])
    for record in json.loads((out / "summary.json").read_text()):
        arm = record["arm"]
        talker_pid = record["pids"]["talker_ar"]
        connection = sqlite3.connect(out / arm / "trace.sqlite")
        names = dict(connection.execute("SELECT id, value FROM StringIds"))
        thread_names = {
            tid: names.get(name_id, "?") for name_id, tid in connection.execute("SELECT nameId, globalTid FROM ThreadNames")
        }
        launches: dict[int, int] = defaultdict(int)
        synchronizes: dict[int, int] = defaultdict(int)
        api_ms: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        window_start, window_end = None, None
        for start, end, tid, name_id in connection.execute(
            "SELECT start, end, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ):
            if process_of(tid) != talker_pid:
                continue
            else:
                pass
            name = names.get(name_id, "")
            window_start = start if window_start is None else min(window_start, start)
            window_end = end if window_end is None else max(window_end, end)
            api_ms[tid][name.split("_v")[0]] += (end - start) / 1e6
            if name.startswith("cudaGraphLaunch") or name.startswith("cudaLaunchKernel"):
                launches[tid] += 1
            elif name.startswith("cudaEventSynchronize"):
                synchronizes[tid] += 1
            else:
                pass
        waits: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for start, end, tid, name_id in connection.execute("SELECT start, end, globalTid, nameId FROM OSRT_API"):
            if process_of(tid) != talker_pid:
                continue
            else:
                pass
            name = names.get(name_id, "")
            if name in WAITS:
                waits[tid][name] += (end - start) / 1e6
            else:
                pass
        window_s = (window_end - window_start) / 1e9 if window_start is not None else 0.0
        print(f"== {arm} (card {record['card']}), talker pid {talker_pid}, window {window_s:.1f} s")
        for tid in sorted(launches, key=lambda tid: -launches[tid])[:2]:
            wait_ms = {name: round(value, 1) for name, value in sorted(waits[tid].items())}
            top_api = sorted(api_ms[tid].items(), key=lambda item: -item[1])[:8]
            print(
                f"  tid {tid & 0xFFFFFF:7d} launches {launches[tid]:7d} event syncs {synchronizes.get(tid, 0):6d} "
                f"waits ms {wait_ms}\n    cuda api ms {[(name, round(value, 1)) for name, value in top_api]}"
            )


if __name__ == "__main__":
    main()
