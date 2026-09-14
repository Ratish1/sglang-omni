"""Interpreter lock waits per thread from an nsys sqlite export.

Usage: python nsys_lock_waits.py <window.sqlite> <threads.json> [--top 6]

On CPython 3.12 a contended interpreter lock acquire is pthread_cond_timedwait and is
never called when the lock is free; Python locks and queues wait in sem_wait and
sem_clockwait. The script takes the threads with the most launches from the
threads.json that nsys_threads.py wrote, and prints per thread the count and summed
ms of each wait kind over the window, with the window length for the percent.
"""

import argparse
import json
import sqlite3

WAITS = ("pthread_cond_timedwait", "sem_wait", "sem_clockwait", "pthread_rwlock_rdlock")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("threads_json")
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    threads = json.load(open(args.threads_json))
    tids = sorted(threads, key=lambda tid: -threads[tid]["launch_n"])[: args.top]
    db = sqlite3.connect(args.sqlite)
    (window_ns,) = db.execute("SELECT MAX(end) - MIN(start) FROM OSRT_API").fetchone()
    names = ", ".join(f"'{name}'" for name in WAITS)
    print(f"window {window_ns / 1e9:.1f} s")
    print(
        f"{'tid':>16s} {'launches':>8s}  " + "  ".join(f"{name:>28s}" for name in WAITS)
    )
    for tid in tids:
        rows = db.execute(
            "SELECT s.value, COUNT(*), SUM(o.end - o.start) FROM OSRT_API o "
            "JOIN StringIds s ON s.id = o.nameId "
            f"WHERE o.globalTid = ? AND s.value IN ({names}) GROUP BY s.value",
            (int(tid),),
        ).fetchall()
        by_name = {name: (count, total) for name, count, total in rows}
        cells = []
        for name in WAITS:
            count, total = by_name.get(name, (0, 0))
            cells.append(
                f"{count:8d} / {total / 1e6:7.0f} ms ({100 * total / window_ns:4.1f}%)"
            )
        print(f"{tid:>16s} {threads[tid]['launch_n']:8d}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
