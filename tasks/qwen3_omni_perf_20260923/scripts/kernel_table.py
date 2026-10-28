"""Per-process kernel table over a census window, graph node kernels included.

For each process (named from the census's process line), the GPUs it ran on and its largest
kernels by name with call counts. Each kernel is charged its exclusive time on its stream:
the part after the previous kernel of that stream ended, since a kernel launched with
programmatic dependent launch starts early and waits inside itself for its predecessor, so
its raw duration repeats the predecessor's. The raw sum is printed beside it. On a card that
several processes share, both still include time the context was switched out.

usage: python kernel_table.py <serve.sqlite> <window.txt> <census.txt> [top]
"""

from __future__ import annotations

import re
import sqlite3
import sys

from nsys_stage_ledger import pid_of, session_window


def main() -> None:
    database = sqlite3.connect(sys.argv[1])
    start, end = session_window(database, sys.argv[2])
    with open(sys.argv[3]) as handle:
        census = handle.read()
    top = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    process_line = next(
        line for line in census.splitlines() if line.startswith("processes: ")
    )
    names = {int(pid): stage for pid, stage in re.findall(r"(\d+)=(\w+)", process_line)}
    strings = dict(database.execute("select id, value from StringIds"))
    rows = database.execute(
        "select globalPid, deviceId, streamId, shortName, start, end"
        " from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and end <= ?"
        " order by globalPid, deviceId, streamId, start",
        (start, end),
    ).fetchall()
    window_ms = (end - start) / 1e6
    per_process: dict[int, dict[str, list[float]]] = {}
    devices: dict[int, set[int]] = {}
    stream_end: dict[tuple[int, int, int], int] = {}
    for global_pid, device, stream, short_name, kernel_start, kernel_end in rows:
        pid = pid_of(global_pid)
        key = (global_pid, device, stream)
        exclusive_start = max(kernel_start, stream_end.get(key, kernel_start))
        stream_end[key] = max(kernel_end, stream_end.get(key, kernel_end))
        kernels = per_process.setdefault(pid, {})
        entry = kernels.setdefault(
            strings.get(short_name, str(short_name)), [0.0, 0.0, 0]
        )
        entry[0] += max(kernel_end - exclusive_start, 0) / 1e6
        entry[1] += (kernel_end - kernel_start) / 1e6
        entry[2] += 1
        devices.setdefault(pid, set()).add(device)
    print(f"window {window_ms:.0f} ms")
    for pid, kernels in sorted(
        per_process.items(), key=lambda item: -sum(v[0] for v in item[1].values())
    ):
        exclusive = sum(value[0] for value in kernels.values())
        raw = sum(value[1] for value in kernels.values())
        count = sum(value[2] for value in kernels.values())
        print(
            f"\n{names.get(pid, pid)} pid={pid} gpu={sorted(devices[pid])} exclusive "
            f"{exclusive:.0f} ms ({100 * exclusive / window_ms:.1f} % of the window), "
            f"raw {raw:.0f} ms, {count} kernels"
        )
        for name, (milliseconds, raw_ms, calls) in sorted(
            kernels.items(), key=lambda item: -item[1][0]
        )[:top]:
            print(
                f"  {100 * milliseconds / exclusive:5.1f} %  {milliseconds:8.1f} ms  "
                f"{calls:7d} x {1e3 * milliseconds / calls:8.1f} us  "
                f"(raw {1e3 * raw_ms / calls:8.1f} us)  {name[:80]}"
            )


if __name__ == "__main__":
    main()
