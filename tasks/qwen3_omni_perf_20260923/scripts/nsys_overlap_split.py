"""Per process kernel time of one nsys export split by whether another process had a kernel
running at the same instant (run on the box), so a slower process under MPS can be told
apart as slowed by concurrent work or slowed alone.

Prints, per process named by the stage its recorder marks carry: kernels, kernel time, the
share of it with another process's kernel running, and for the top kernel names the mean
duration of launches that ran alone against launches that overlapped another process.

usage: python nsys_overlap_split.py REPORT.sqlite [--window window.txt] [--top 12]
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import statistics

from nsys_ctx_slices import merged, overlap_ns
from nsys_stage_ledger import nvtx_rows, pid_of, session_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    t0, t1 = session_window(db, args.window)

    stage_votes: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for _, _, global_tid, annotation in nvtx_rows(db, t0, t1):
        stage = annotation.get("stage", "unknown")
        if annotation.get("kind") == "mark" and stage != "unknown":
            stage_votes[pid_of(global_tid)][stage] += 1
        else:
            pass
    stage_by_pid = {
        pid: votes.most_common(1)[0][0] for pid, votes in stage_votes.items()
    }

    names = dict(db.execute("select id, value from StringIds"))
    kernels: dict[int, list[tuple[int, int, str]]] = collections.defaultdict(list)
    for start, end, global_pid, short in db.execute(
        "select start, end, globalPid, shortName from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?",
        (t0, t1),
    ):
        kernels[pid_of(global_pid)].append((start, end, names[short]))
    unions = {
        pid: merged([(start, end) for start, end, _ in rows])
        for pid, rows in kernels.items()
    }

    for pid, rows in sorted(kernels.items(), key=lambda item: -len(item[1])):
        others = merged(
            [span for other, spans in unions.items() if other != pid for span in spans]
        )
        others_starts = [start for start, _ in others]
        total = overlapped = 0
        alone: dict[str, list[float]] = collections.defaultdict(list)
        shared: dict[str, list[float]] = collections.defaultdict(list)
        for start, end, name in rows:
            covered = overlap_ns(others, others_starts, start, end)
            total += end - start
            overlapped += covered
            if covered == 0:
                alone[name].append((end - start) / 1e3)
            else:
                shared[name].append((end - start) / 1e3)
        print(
            f"{stage_by_pid.get(pid, '?')} ({pid}) kernels {len(rows)} kernel ms {total / 1e6:.1f} "
            f"overlapped by another process {overlapped / 1e6:.1f} ms ({overlapped / max(total, 1):.1%})"
        )
        by_time = collections.Counter()
        for start, end, name in rows:
            by_time[name] += end - start
        for name, _ in by_time.most_common(args.top):
            a, s = alone.get(name, []), shared.get(name, [])
            mean_a = f"{statistics.fmean(a):8.2f}" if a else "       -"
            mean_s = f"{statistics.fmean(s):8.2f}" if s else "       -"
            print(
                f"  alone {len(a):7d} mean us {mean_a}  overlapped {len(s):7d} mean us {mean_s}  {name[:60]}"
            )


if __name__ == "__main__":
    main()
