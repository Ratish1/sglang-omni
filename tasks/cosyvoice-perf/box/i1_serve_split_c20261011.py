# SPDX-License-Identifier: Apache-2.0
"""Item 1 pairs: the AR and the vocoder per cell from each arm's serve.log, split by the cell times in the
arm's progress.txt. AR: SGLang's decode log lines (one per 40 decode steps) give the running requests and the
generation throughput, so the throughput per running request and the decode steps per second. Vocoder: the
Flow batch lines give the steps and rows per step by kind (causal hop, leftover final).

    python3 i1_serve_split_c20261011.py /data/c20261011/i1
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

DECODE = re.compile(r"Decode batch, #running-req: (\d+), .* gen throughput \(token/s\): ([\d.]+)")
FLOW = re.compile(r"Fun-CosyVoice3 (\w+) Flow batch size=(\d+)")
CELL = re.compile(r"cell (\w+) rc \d+ (\d\d:\d\d:\d\d)")


def clock(text: str) -> datetime:
    return datetime.strptime(text, "%H:%M:%S")


def main() -> None:
    root = Path(sys.argv[1])
    for pair in ("a", "b"):
        for arm in sorted((root / pair).glob("card*")):
            progress = (arm / "progress.txt").read_text()
            start = clock(re.search(r"healthy (\d\d:\d\d:\d\d)", progress).group(1))
            windows = []
            for name, end in CELL.findall(progress):
                windows.append((name, start, clock(end)))
                start = clock(end)
            stats = {name: {"decode": [], "flow": {}} for name, _, _ in windows}
            for line in (arm / "serve.log").read_text().splitlines():
                if len(line) < 19 or not line[:4].isdigit():
                    continue
                when = clock(line[11:19])
                cell = next((name for name, begin, end in windows if begin <= when < end), None)
                if cell is None:
                    continue
                if match := DECODE.search(line):
                    stats[cell]["decode"].append((int(match.group(1)), float(match.group(2))))
                elif match := FLOW.search(line):
                    stats[cell]["flow"].setdefault(match.group(1), []).append(int(match.group(2)))
            for name, begin, end in windows:
                decode = stats[name]["decode"]
                running = statistics.fmean(row for row, _ in decode)
                throughput = statistics.fmean(rate for _, rate in decode)
                per_request = statistics.fmean(rate / row for row, rate in decode)
                steps_per_s = statistics.fmean(rate / row for row, rate in decode if row > 0)
                flow = " ".join(
                    f"{kind} {len(sizes)} of {statistics.fmean(sizes):.2f}"
                    for kind, sizes in sorted(stats[name]["flow"].items())
                )
                print(
                    f"{pair} {arm.name:12s} {name:4s} {(end - begin).seconds:3d} s | decode lines {len(decode):4d}"
                    f" running {running:5.2f} gen {throughput:7.1f} tok/s per request {per_request:6.1f}"
                    f" (steps/s {steps_per_s:6.1f}) | flow {flow}"
                )


if __name__ == "__main__":
    main()
