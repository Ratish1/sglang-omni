"""Startup and serving memory of a memory_run.sh arm: torch live and reserved per vocoder step
(count probe), card memory from nvidia-smi dmon, and the cells' throughput.
usage: python3 memory_summary.py <arm dir>..."""

import json
import re
import sys
from pathlib import Path

STEP = re.compile(
    r"voc\.(?:step|batch) t=([\d.]+) (?:plan=(\S+) )?rows=(\d+) ms=([\d.]+) "
    r"alloc_gib=([\d.]+) peak_gib=([\d.]+) reserved_gib=([\d.]+)"
)


def card_mib(arm: Path) -> list[int]:
    header = None
    values = []
    for line in (arm / "dmon.log").read_text().splitlines():
        if line.startswith("#"):
            header = header or line.lstrip("#").split()
            continue
        parts = line.split()
        if header and len(parts) == len(header) and parts[header.index("fb")].isdigit():
            values.append(int(parts[header.index("fb")]))
    return values


for arm in map(Path, sys.argv[1:]):
    steps = [STEP.search(line) for line in (arm / "serve.log").read_text().splitlines()]
    steps = [m for m in steps if m]
    alloc = [float(m.group(5)) for m in steps]
    peak = [float(m.group(6)) for m in steps]
    reserved = [float(m.group(7)) for m in steps]
    transient = max(p - a for p, a in zip(peak, alloc))
    fb = card_mib(arm)
    serving_fb = [v for v in fb if v > 0.9 * max(fb)]
    cells = {}
    for cell in ("s64", "b32", "L16"):
        summary = json.loads((arm / cell / "speed_results.json").read_text())["summary"]
        cells[cell] = (summary["throughput_qps"], summary["failed_requests"])
    print(
        f"{arm.name}: steps={len(steps)} live min {min(alloc):.3f} GiB, live transient max "
        f"{transient:.3f}, reserved first {reserved[0]:.3f} max {max(reserved):.3f} "
        f"(+{max(reserved) - reserved[0]:.3f}); card MiB at first step band {serving_fb[0]} "
        f"max {max(fb)} (+{(max(fb) - serving_fb[0]) / 1024:.2f} GiB); "
        + ", ".join(f"{c} {q:.3f} req/s failed {f}" for c, (q, f) in cells.items())
    )
