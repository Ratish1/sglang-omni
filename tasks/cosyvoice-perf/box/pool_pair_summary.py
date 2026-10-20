"""Prefix pool size, card memory at healthy and at peak, the probe's live transient and reserved
growth, and the prefix cache's hit share per cell, for each arm of an item 3 pair.
usage: python3 pool_pair_summary.py <arm dir>..."""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

STEP = re.compile(
    r"voc\.(?:step|batch) t=([\d.]+) .*?alloc_gib=([\d.]+) peak_gib=([\d.]+) reserved_gib=([\d.]+)"
)
PREFIX = re.compile(r"prefix\.step t=([\d.]+) rows=(\d+) cached=(\d+) plain=(\d+)")


def seconds(hms: str) -> int:
    hours, minutes, secs = map(int, hms.split(":"))
    return hours * 3600 + minutes * 60 + secs


def epoch(day: str, hms: str) -> float:
    return datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


for arm in map(Path, sys.argv[1:]):
    progress = (arm / "progress.txt").read_text()
    start = re.search(r"start (\d+:\d+:\d+)", progress).group(1)
    healthy = re.search(r"healthy (\d+:\d+:\d+)", progress).group(1)
    cells = re.findall(r"cell (\S+) rc \d+ (\d+:\d+:\d+)", progress)
    log = (arm / "serve.log").read_text()
    pool = re.search(r"Flow prefix cache: (\d+) frames \(([\d.]+) GB", log)
    header = None
    card = []
    for line in (arm / "dmon.log").read_text().splitlines():
        if line.startswith("#"):
            header = header or line.lstrip("#").split()
            continue
        parts = line.split()
        if header and len(parts) == len(header) and parts[header.index("fb")].isdigit():
            card.append(int(parts[header.index("fb")]))
    at_healthy = card[seconds(healthy) - seconds(start)]
    steps = [tuple(map(float, m.groups())) for m in STEP.finditer(log)]
    print(
        f"{arm}: pool {pool.group(2) if pool else 'none'} GB ({pool.group(1) if pool else 0} frames), "
        f"card MiB healthy {at_healthy} peak {max(card)} (+{(max(card) - at_healthy) / 1024:.2f} GiB), "
        f"live transient max {max(p - a for _, a, p, _ in steps):.3f} GiB, "
        f"reserved {steps[0][3]:.2f} -> {max(r for *_, r in steps):.2f} GiB"
    )
    day = re.search(r"(\d{4}-\d{2}-\d{2}) \d+:\d+:\d+", log).group(1)
    previous = epoch(day, healthy)
    for name, end in cells:
        window = [m for m in PREFIX.finditer(log) if previous <= float(m.group(1)) < epoch(day, end)]
        rows = sum(int(m.group(2)) for m in window)
        cached = sum(int(m.group(3)) for m in window)
        print(f"   {name}: causal rows {rows}, cached {cached / max(rows, 1) * 100:.1f} %")
        previous = epoch(day, end)
