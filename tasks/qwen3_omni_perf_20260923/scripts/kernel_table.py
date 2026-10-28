"""Per-process kernel table over a census window, graph node kernels included.

For each process (named from the census's process line), the summed kernel time, the GPUs it
ran on, and its largest kernels by name with call counts and mean duration. Sums are over
kernel durations, so a process's total is its card time, not a union, and processes on
different cards are read side by side.

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
        "select globalPid, deviceId, shortName, start, end from CUPTI_ACTIVITY_KIND_KERNEL"
        " where start >= ? and end <= ?",
        (start, end),
    ).fetchall()
    window_ms = (end - start) / 1e6
    per_process: dict[int, dict[str, list[float]]] = {}
    devices: dict[int, set[int]] = {}
    for global_pid, device, short_name, kernel_start, kernel_end in rows:
        pid = pid_of(global_pid)
        kernels = per_process.setdefault(pid, {})
        entry = kernels.setdefault(strings.get(short_name, str(short_name)), [0.0, 0])
        entry[0] += (kernel_end - kernel_start) / 1e6
        entry[1] += 1
        devices.setdefault(pid, set()).add(device)
    print(f"window {window_ms:.0f} ms")
    for pid, kernels in sorted(
        per_process.items(), key=lambda item: -sum(v[0] for v in item[1].values())
    ):
        total = sum(value[0] for value in kernels.values())
        count = sum(value[1] for value in kernels.values())
        print(
            f"\n{names.get(pid, pid)} pid={pid} gpu={sorted(devices[pid])} "
            f"kernel time {total:.0f} ms ({100 * total / window_ms:.1f} % of the window), "
            f"{count} kernels"
        )
        for name, (milliseconds, calls) in sorted(
            kernels.items(), key=lambda item: -item[1][0]
        )[:top]:
            print(
                f"  {100 * milliseconds / total:5.1f} %  {milliseconds:8.1f} ms  "
                f"{calls:7d} x {1e3 * milliseconds / calls:8.1f} us  {name[:90]}"
            )


if __name__ == "__main__":
    main()
