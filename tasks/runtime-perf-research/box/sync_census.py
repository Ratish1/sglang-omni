"""A stage's blocking runtime calls in an nsys trace: count, total and median ms per call name, per graph replay.
usage: python3 - TRACE.sqlite SERVE.LOG STAGE < sync_census.py"""

import re
import sqlite3
import statistics
import sys
from collections import defaultdict

STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=\d+ .*pid=(\d+)")


def main() -> None:
    trace, serve_log, stage = sys.argv[1], sys.argv[2], sys.argv[3]
    stages = {}
    with open(serve_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := STARTUP.search(line):
                stages[int(match[2])] = match[1]
    pid = next(p for p, s in stages.items() if s == stage)
    db = sqlite3.connect(trace)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    first, last = db.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
    lo, hi = first + 1_000_000_000, last - 1_000_000_000
    by_name = defaultdict(list)
    by_thread = defaultdict(int)
    for start, end, name_id, tid in db.execute(
        "SELECT start, end, nameId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE ((globalTid >> 24) & 16777215) = ?",
        (pid,),
    ):
        if start < lo or end > hi:
            continue
        else:
            pass
        name = f"{tid & 0xFFFFFF} {strings.get(name_id, str(name_id))}"
        by_name[name].append((end - start) / 1e6)
        by_thread[tid & 0xFFFFFF] += end - start
    replays = db.execute(
        "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE ((globalPid >> 24) & 16777215) = ? AND start >= ? AND end <= ?",
        (pid, lo, hi),
    ).fetchone()[0]
    print(f"{stage} pid {pid}, window {(hi - lo) / 1e9:.2f} s, {replays} graph replays")
    print("| call | count | per replay | total ms | median ms | p90 ms |")
    print("|---|---|---|---|---|---|")
    for name, values in sorted(by_name.items(), key=lambda item: -sum(item[1]))[:14]:
        ordered = sorted(values)
        print(f"| {name[:40]} | {len(values)} | {len(values) / max(replays, 1):.2f} | {sum(values):.0f} | "
              f"{statistics.median(ordered):.3f} | {ordered[int(0.9 * len(ordered))]:.3f} |")
    print("threads by runtime ms:", ", ".join(f"{tid} {value / 1e6:.0f}" for tid, value in
                                               sorted(by_thread.items(), key=lambda item: -item[1])[:4]))


if __name__ == "__main__":
    main()
