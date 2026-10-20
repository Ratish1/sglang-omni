# SPDX-License-Identifier: Apache-2.0
"""E2 analysis: from one server's probe lines (streams.<pid>.jsonl, the process with the most device
memory) and its progress.txt, per (stream, segment pool, small or large): mapped segment bytes at
healthy and their peak over each cell and the whole run; the same for the memory outside torch, torch
reserved and used device memory. Times are local clock seconds, progress.txt lines are HH:MM:SS of the
same day.

usage: python e2_streams_c20261011.py <arm dir>
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

GIB = 2**30


def clock_seconds(stamp: str, day: time.struct_time) -> float:
    hours, minutes, seconds = (int(part) for part in stamp.split(":"))
    return time.mktime(
        (day.tm_year, day.tm_mon, day.tm_mday, hours, minutes, seconds, 0, 0, -1)
    )


def main() -> None:
    arm = sys.argv[1]
    files = glob.glob(os.path.join(arm, "streams.*.jsonl"))
    samples_by_file = {
        path: [json.loads(line) for line in open(path) if line.strip()] for path in files
    }
    path, samples = max(
        samples_by_file.items(), key=lambda item: max(s["used"] for s in item[1])
    )
    day = time.localtime(samples[0]["t"])
    marks: list[tuple[str, float]] = []
    for line in open(os.path.join(arm, "progress.txt")):
        words = line.split()
        if words and words[0] == "healthy":
            marks.append(("healthy", clock_seconds(words[1], day)))
        elif words and words[0] == "cell":
            marks.append((words[1], clock_seconds(words[-1], day)))
    healthy_t = next(t for name, t in marks if name == "healthy")
    windows = [
        (name, start, end)
        for (_, start), (name, end) in zip(marks, marks[1:], strict=False)
    ]
    at_healthy = min(samples, key=lambda s: abs(s["t"] - healthy_t))
    keys = sorted({key for s in samples for key in s["per"]})
    print(f"process file {os.path.basename(path)}, {len(samples)} samples")
    header = ["total", "torch_reserved", "non_torch"]
    print("bytes in GiB: at healthy, then peak per cell, then growth of the run's peak over healthy")
    print("| series | healthy | " + " | ".join(name for name, _, _ in windows) + " | growth |")
    print("| --- | --- | " + " | ".join("---" for _ in windows) + " | --- |")

    def series(value) -> None:
        name, getter = value
        row = [f"{getter(at_healthy) / GIB:.3f}"]
        run_peak = getter(at_healthy)
        for _, start, end in windows:
            inside = [getter(s) for s in samples if start <= s["t"] <= end]
            peak = max(inside) if inside else float("nan")
            run_peak = max(run_peak, peak) if inside else run_peak
            row.append(f"{peak / GIB:.3f}")
        row.append(f"{(run_peak - getter(at_healthy)) / GIB:+.3f}")
        print(f"| {name} | " + " | ".join(row) + " |")

    series(("used", lambda s: s["used"]))
    for name in header[1:]:
        series((name, lambda s, name=name: s[name]))
    for key in keys:
        series((f"segments {key}", lambda s, key=key: s["per"].get(key, [0, 0])[0]))


if __name__ == "__main__":
    main()
