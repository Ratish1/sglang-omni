#!/usr/bin/env python3
"""Calculate a union of captured CUDA-kernel intervals, NOT SM utilization.
Python 3.10+, standard library only. Nsight SQLite schemas vary: validate first.
Require an explicit analysis window to avoid hiding pre/post-kernel idle time.
Timestamps are the exported SQLite clock in nanoseconds, not host wall time.
"""
from __future__ import annotations
import argparse
import heapq
import json
from pathlib import Path
import sqlite3
import sys
from typing import Iterable

CANDIDATES = ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL')


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def summarize(intervals: Iterable[tuple[int, int]], start_ns: int, end_ns: int,
              top_gaps: int = 20) -> dict:
    """Input intervals must be ordered by start time. Arithmetic stays integer."""
    if end_ns <= start_ns:
        raise ValueError('end_ns must exceed start_ns')
    if top_gaps < 1:
        raise ValueError('top_gaps must be positive')
    cursor = start_ns
    kernel_count = invalid_count = union_ns = raw_ns = gap_count = 0
    maximum_gap_ns = 0
    previous_start = None
    largest: list[tuple[int, int, int]] = []

    def add_gap(a: int, b: int) -> None:
        nonlocal gap_count, maximum_gap_ns
        if b <= a:
            return
        gap_count += 1
        duration = b - a
        maximum_gap_ns = max(maximum_gap_ns, duration)
        item = (duration, a, b)
        if len(largest) < top_gaps:
            heapq.heappush(largest, item)
        elif item > largest[0]:
            heapq.heapreplace(largest, item)

    for s, e in intervals:
        s, e = int(s), int(e)
        if previous_start is not None and s < previous_start:
            raise ValueError('intervals are not ordered by start time')
        previous_start = s
        if e <= s:
            invalid_count += 1
            continue
        a, b = max(s, start_ns), min(e, end_ns)
        if b <= a:
            continue
        kernel_count += 1
        raw_ns += b - a
        if a > cursor:
            add_gap(cursor, a)
        if b > cursor:
            union_ns += b - max(a, cursor)
            cursor = b
    add_gap(cursor, end_ns)
    window_ns = end_ns - start_ns
    return {
        'window_start_ns': start_ns, 'window_end_ns': end_ns,
        'window_seconds': window_ns / 1e9,
        'captured_kernel_count_intersecting_window': kernel_count,
        'invalid_intervals_ignored': invalid_count,
        'captured_kernel_union_ns': union_ns,
        'captured_kernel_union_fraction': union_ns / window_ns,
        'captured_kernel_union_percent': 100.0 * union_ns / window_ns,
        'sum_of_clipped_kernel_durations_ns': raw_ns,
        'sum_over_union_overlap_factor': raw_ns / union_ns if union_ns else None,
        'no_captured_kernel_ns': window_ns - union_ns,
        'gap_count_including_window_edges': gap_count,
        'maximum_gap_ms': maximum_gap_ns / 1e6,
        'largest_no_captured_kernel_intervals': [
            {'start_ns': a, 'end_ns': b, 'duration_ms': d / 1e6,
             'touches_window_edge': a == start_ns or b == end_ns}
            for d, a, b in sorted(largest, reverse=True)
        ],
        'limitations': [
            'This is captured CUDA-kernel timeline coverage, NOT SM activity, occupancy, FLOP efficiency, or physical-device utilization.',
            'Uncaptured processes, child workers, graph nodes, dropped events, and non-CUDA work can make coverage incomplete.',
            'A no-kernel interval can contain copies, CPU work, dependencies, or untraced work; it is not automatically wasted time.',
            'Separate physical devices and choose a steady-state window including its idle intervals.',
            'Do not add overlapping kernel durations to compute duty cycle.'
        ]
    }


def open_database(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise ValueError(f'file not found: {path}')
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    conn.execute('PRAGMA query_only = ON')
    return conn


def choose_table(conn: sqlite3.Connection, requested: str | None) -> tuple[str, dict[str, str]]:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if requested:
        if requested not in tables:
            raise ValueError(f'table {requested!r} not found; available kernel tables: {sorted(t for t in tables if "KERNEL" in t)}')
        table = requested
    else:
        candidates = [t for t in CANDIDATES if t in tables]
        populated = [t for t in candidates if conn.execute(f'SELECT 1 FROM {quote_ident(t)} LIMIT 1').fetchone()]
        if len(populated) > 1:
            raise ValueError(f'multiple populated kernel tables {populated}; inspect and choose --kernel-table explicitly')
        if not populated:
            raise ValueError('no populated supported CUDA kernel table; check CUDA tracing, child-worker coverage, graph tracing, and export schema')
        table = populated[0]
    columns = {row[1].lower(): row[1] for row in conn.execute(f'PRAGMA table_info({quote_ident(table)})')}
    missing = {'start', 'end', 'deviceid'} - columns.keys()
    if missing:
        raise ValueError(f'unsupported schema: missing {sorted(missing)} in {table}; columns={list(columns.values())}')
    return table, columns


def devices(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> list[dict]:
    d, s, e = (quote_ident(columns[k]) for k in ('deviceid', 'start', 'end'))
    sql = f'SELECT {d}, COUNT(*), MIN({s}), MAX({e}) FROM {quote_ident(table)} GROUP BY {d} ORDER BY {d}'
    return [{'export_device_id': x[0], 'kernel_count': x[1],
             'first_kernel_start_ns': x[2], 'last_kernel_end_ns': x[3]}
            for x in conn.execute(sql)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--kernel-table')
    parser.add_argument('--device', type=int)
    parser.add_argument('--start-ns', type=int)
    parser.add_argument('--end-ns', type=int)
    parser.add_argument('--top-gaps', type=int, default=20)
    args = parser.parse_args()
    try:
        with open_database(args.database) as conn:
            table, cols = choose_table(conn, args.kernel_table)
            catalogue = devices(conn, table, cols)
            if args.describe:
                result = {'database': str(args.database.resolve()), 'kernel_table': table,
                          'columns': list(cols.values()), 'devices': catalogue,
                          'note': 'These bounds omit outer idle time. Choose explicit steady-state start/end from the exported timeline; do not silently substitute these bounds.'}
            else:
                if None in (args.device, args.start_ns, args.end_ns):
                    raise ValueError('specify --device, --start-ns, --end-ns, or use --describe')
                if args.device not in {x['export_device_id'] for x in catalogue}:
                    raise ValueError(f'export device {args.device} not found; available: {catalogue}')
                d, s, e = (quote_ident(cols[k]) for k in ('deviceid', 'start', 'end'))
                sql = (f'SELECT {s}, {e} FROM {quote_ident(table)} '
                       f'WHERE {d}=? AND {e}>? AND {s}<? ORDER BY {s}, {e}')
                rows = conn.execute(sql, (args.device, args.start_ns, args.end_ns))
                result = summarize(rows, args.start_ns, args.end_ns, args.top_gaps)
                result.update({'database': str(args.database.resolve()),
                               'kernel_table': table, 'export_device_id': args.device})
        print(json.dumps(result, indent=2))
    except (ValueError, sqlite3.Error, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
