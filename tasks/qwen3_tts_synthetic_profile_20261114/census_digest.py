"""Step and vocoder batch census of one served timing cell (nsys SQLite export, probe labels on):
talker steps by mode, vocoder initial and follow-up batches by stream count, and kernel time per
CUDA stream, all inside the window from the first to the last labeled talker step.

Usage: python3 census_digest.py <cell.sqlite>
"""

import argparse
import re
import sqlite3
from collections import defaultdict

STEP = re.compile(r"step\[(\w+) bs=(\d+)")
VOCODER = re.compile(r"vocoder\.(initial|followup)\[streams=(\d+)\]")


def union_ms(intervals: list[tuple[int, int]], start: int, end: int) -> float:
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals if b > start and a < end)
    total = 0
    cursor = start
    for a, b in clipped:
        a = max(a, cursor)
        if b > a:
            total += b - a
            cursor = b
        else:
            pass
    return total / 1e6


def print_step_split(connection: sqlite3.Connection, window_start: int, window_end: int) -> None:
    """Per mode and batch size: step time, GPU time of the step thread's own launches, and OS waits."""
    tables = {row[0] for row in connection.execute("select name from sqlite_master where type = 'table'")}
    steps = [
        (start, end, tid, text)
        for start, end, tid, text in connection.execute(
            """
            select e.start, e.end, e.globalTid, coalesce(s.value, e.text, '')
            from NVTX_EVENTS e left join StringIds s on s.id = e.textId
            where e.end is not null and e.end > e.start
            """
        )
        if STEP.search(text)
    ]
    step_threads = {tid for _, _, tid, _ in steps}
    launches: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for activity in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if activity in tables:
            for tid, start, end in connection.execute(
                f"""
                select r.globalTid, a.start, a.end from {activity} a
                join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = a.correlationId
                where a.end > ? and a.start < ?
                """,
                (window_start, window_end),
            ):
                if tid in step_threads:
                    launches[tid].append((start, end))
                else:
                    pass
        else:
            pass
    waits: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if "OSRT_API" in tables:
        for tid, start, end in connection.execute(
            "select globalTid, start, end from OSRT_API where end > ? and start < ?",
            (window_start, window_end),
        ):
            if tid in step_threads:
                waits[tid].append((start, end))
            else:
                pass
    else:
        pass
    split: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    for start, end, tid, text in steps:
        mode, batch = STEP.search(text).groups()
        split[(mode, int(batch))].append(
            ((end - start) / 1e6, union_ms(launches[tid], start, end), union_ms(waits[tid], start, end))
        )
    print("talker steps by mode and batch size: step ms, own GPU ms, OS wait ms")
    for (mode, batch), rows in sorted(split.items()):
        count = len(rows)
        step_ms, gpu_ms, wait_ms = (sum(row[index] for row in rows) / count for index in range(3))
        print(f"  {mode} bs={batch}: {count} steps, step {step_ms:.2f}, gpu {gpu_ms:.2f}, os wait {wait_ms:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    args = parser.parse_args()
    connection = sqlite3.connect(args.sqlite)
    ranges = connection.execute(
        """
        select e.start, e.end, coalesce(s.value, e.text, '')
        from NVTX_EVENTS e left join StringIds s on s.id = e.textId
        where e.end is not null and e.end > e.start
        """
    ).fetchall()
    steps = [(start, end, text) for start, end, text in ranges if STEP.search(text)]
    window_start = min(start for start, _, _ in steps)
    window_end = max(end for _, end, _ in steps)
    window_ms = (window_end - window_start) / 1e6
    print(f"window {window_ms:.1f} ms, talker steps {len(steps)}, {window_ms / len(steps):.2f} ms a step")
    by_mode: dict[str, list[float]] = defaultdict(list)
    for start, end, text in steps:
        mode, _ = STEP.search(text).groups()
        by_mode[mode].append((end - start) / 1e6)
    for mode, durations in sorted(by_mode.items()):
        print(f"  {mode}: {len(durations)} steps, mean {sum(durations) / len(durations):.2f} ms")
    batches: dict[tuple[str, int], list[float]] = defaultdict(list)
    for start, end, text in ranges:
        match = VOCODER.search(text)
        if match and window_start <= start <= window_end:
            batches[(match.group(1), int(match.group(2)))].append((end - start) / 1e6)
        else:
            pass
    for kind in ("initial", "followup"):
        rows = sorted((streams, durations) for (name, streams), durations in batches.items() if name == kind)
        count = sum(len(durations) for _, durations in rows)
        streams_total = sum(streams * len(durations) for streams, durations in rows)
        busy = sum(sum(durations) for _, durations in rows)
        print(f"vocoder {kind}: {count} batches, {streams_total} streams, {busy:.1f} ms in batches")
        for streams, durations in rows:
            print(f"  streams={streams}: {len(durations)} batches, mean {sum(durations) / len(durations):.3f} ms")
    print_step_split(connection, window_start, window_end)
    kernels = connection.execute(
        "select streamId, start, end from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?",
        (window_start, window_end),
    ).fetchall()
    per_stream: dict[int, float] = defaultdict(float)
    for stream, start, end in kernels:
        per_stream[stream] += (min(end, window_end) - max(start, window_start)) / 1e6
    print("kernel time per stream in the window:")
    for stream, busy in sorted(per_stream.items(), key=lambda item: -item[1]):
        print(f"  stream {stream}: {busy:.1f} ms ({busy / window_ms:.1%})")


if __name__ == "__main__":
    main()
