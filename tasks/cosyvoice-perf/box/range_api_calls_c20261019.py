"""CUDA runtime and driver calls made inside one NVTX range on the thread that pushed it, from one nsys report: per
call name, the count per range, the time per range and the per call p50, p99 and max, so a call that waits (a
pageable copy waiting for its staging buffer, an allocator call) shows by its duration even when it is not a
synchronizing call.

usage: python3 range_api_calls_c20261019.py NAME=REPORT.sqlite [NAME=REPORT.sqlite ...] --range flow.pack (a range is
matched by its whole text or by its text up to a space, so "voc.step leftover" matches "voc.step leftover rows=4")
"""

from __future__ import annotations

import argparse
import bisect
import sqlite3
from collections import defaultdict

MS = 1e-6


def ranges(db: sqlite3.Connection, label: str) -> dict[int, list[tuple[int, int]]]:
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text = "COALESCE(e.text, s.value)" if "textId" in columns else "e.text"
    join = "LEFT JOIN StringIds s ON e.textId = s.id" if "textId" in columns else ""
    by_thread: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for start, end, tid in db.execute(
        f"SELECT e.start, e.end, e.globalTid FROM NVTX_EVENTS e {join} WHERE ({text} = ? OR {text} LIKE ?) "
        "AND e.end IS NOT NULL",
        (label, f"{label} %"),
    ):
        by_thread[tid].append((start, end))
    for spans in by_thread.values():
        spans.sort()
    return by_thread


def calls(db: sqlite3.Connection, tids: set[int]) -> list[tuple[int, int, int, str, int]]:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    found = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        found.extend(
            row
            for row in db.execute(
                f"SELECT c.start, c.end, c.globalTid, s.value, c.correlationId FROM {table} c "
                "JOIN StringIds s ON c.nameId = s.id"
            )
            if row[2] in tids
        )
    return found


def copies(db: sqlite3.Connection) -> dict[int, tuple[int, int, int, int, int]]:
    return {
        row[0]: row[1:]
        for row in db.execute(
            "SELECT correlationId, bytes, copyKind, srcKind, dstKind, start FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        )
    }


def analyse(name: str, path: str, label: str) -> None:
    db = sqlite3.connect(path)
    spans = ranges(db, label)
    count = sum(len(v) for v in spans.values())
    if not count:
        print(f"\n# {name}: no {label} range")
        return
    starts = {tid: [s for s, _ in v] for tid, v in spans.items()}
    durations: dict[str, list[float]] = defaultdict(list)
    copy_rows = copies(db)
    by_copy: dict[tuple[int, int, int, str], list[float]] = defaultdict(list)
    for start, end, tid, call, correlation in calls(db, set(spans)):
        index = bisect.bisect_right(starts[tid], start) - 1
        if index >= 0 and start < spans[tid][index][1]:
            durations[call].append((end - start) * MS)
            if correlation in copy_rows:
                size, kind, src, dst, _ = copy_rows[correlation]
                bucket = "<64K" if size < 65536 else "<1M" if size < 1 << 20 else ">=1M"
                by_copy[(kind, src, dst, bucket)].append((end - start) * MS)
    range_ms = sum((e - s) * MS for v in spans.values() for s, e in v)
    print(f"\n# {name}: {path}\n{label}: {count} ranges, {range_ms / count:.2f} ms per range")
    print(f"{'call':44} {'per range':>9} {'ms/range':>9} {'p50 ms':>8} {'p99 ms':>8} {'max ms':>8}")
    for call, values in sorted(durations.items(), key=lambda item: -sum(item[1]))[:14]:
        values.sort()
        print(
            f"{call[:44]:44} {len(values) / count:9.1f} {sum(values) / count:9.3f} {values[len(values) // 2]:8.3f} "
            f"{values[int(0.99 * (len(values) - 1))]:8.3f} {values[-1]:8.3f}"
        )
    print(f"copies by (copyKind, srcKind, dstKind, bytes): calls per range, call ms per range, p50, p99, max")
    for key, values in sorted(by_copy.items(), key=lambda item: -sum(item[1])):
        values.sort()
        print(
            f"  {str(key):42} {len(values) / count:9.1f} {sum(values) / count:9.3f} {values[len(values) // 2]:8.3f} "
            f"{values[int(0.99 * (len(values) - 1))]:8.3f} {values[-1]:8.3f}"
        )


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+")
parser.add_argument("--range", required=True)
arguments = parser.parse_args()
for report in arguments.reports:
    report_name, _, report_path = report.partition("=")
    analyse(report_name, report_path, arguments.range)
