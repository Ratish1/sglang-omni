"""Window means of DCGM counters over a benchmark's timed requests.

Reads a dcgm_sample.sh log (host epoch, then dcgmi dmon columns) and the bench log's
"Benchmarking N requests" and "Results saved" stamps (container local time, the same
clock). Every sample in the window counts, zeros included, as nsys GPU metrics do.

usage: python dcgm_window.py dcgm.log bench.log
"""

from __future__ import annotations

import re
import statistics
import sys
from datetime import datetime

NAMES = {
    "GRACT": "GR engine active",
    "SMACT": "SM active",
    "SMOCC": "SM occupancy",
    "TENSO": "tensor pipe active",
    "DRAMA": "DRAM active",
    "FP32A": "FP32 pipe active",
    "FP16A": "FP16 pipe active",
    "THMMA": "tensor HMMA active",
    "INTAC": "integer pipe active",
}
STAMP = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.](\d{3})"


def bench_window(path: str) -> tuple[float, float]:
    start = end = None
    with open(path, errors="replace") as handle:
        lines = handle.readlines()
    for line in lines:
        begun = re.search(STAMP + r".*Benchmarking \d+ requests", line)
        saved = re.search(STAMP + r".*Results saved to", line)
        if begun:
            start = datetime.fromisoformat(
                f"{begun.group(1)}.{begun.group(2)}"
            ).timestamp()
        elif saved and start is not None and end is None:
            end = datetime.fromisoformat(
                f"{saved.group(1)}.{saved.group(2)}"
            ).timestamp()
    if start is None or end is None:
        raise SystemExit(f"no bench window in {path}")
    return start, end


def main() -> None:
    dcgm_log, bench_log = sys.argv[1], sys.argv[2]
    t0, t1 = bench_window(bench_log)
    header, rows = None, []
    with open(dcgm_log) as handle:
        lines = handle.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[1] == "#Entity":
            header = parts[2:]
            continue
        if parts[1] != "GPU" or header is None:
            continue
        stamp = float(parts[0])
        try:
            values = [float(v) for v in parts[3 : 3 + len(header)]]
        except ValueError:
            continue
        if t0 <= stamp <= t1:
            rows.append(values)
    print(f"window {t1 - t0:.1f} s, {len(rows)} samples")
    for index, name in enumerate(header or []):
        values = sorted(r[index] for r in rows)
        if not values:
            continue

        def p(q, values=values):
            return values[min(len(values) - 1, int(q * len(values)))]

        print(
            f"  {NAMES.get(name, name):<22} mean {100 * statistics.fmean(values):6.1f}%"
            f"  p10 {100 * p(0.1):6.1f}%  p50 {100 * p(0.5):6.1f}%  p90 {100 * p(0.9):6.1f}%"
        )


if __name__ == "__main__":
    main()
