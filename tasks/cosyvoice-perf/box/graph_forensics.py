"""Why solve graph steps miss, per benchmark cell of a graph_hits probe boot (ab_boot.sh arm
directory): hit rate by steps and by rows; for prefix steps the rows that are a stream's first hop
(an empty cache, so the row's new frames hold its prompt) and the largest first and later hop rows;
for every miss, how many groups of rows within the top tier and the row slots it splits into, and
the rows that alone pass the top tier. Startup warmups (before healthy) are left out.
usage: python3 graph_forensics.py <arm dir> [row slots, default 16]"""

import datetime
import re
import sys
from pathlib import Path

arm = Path(sys.argv[1])
row_slots = int(sys.argv[2]) if len(sys.argv) > 2 else 16
progress = (arm / "progress.txt").read_text().splitlines()
day = datetime.datetime.fromtimestamp((arm / "progress.txt").stat().st_mtime, datetime.timezone.utc).date()


def epoch(clock: str) -> float:
    return datetime.datetime.combine(day, datetime.time.fromisoformat(clock), datetime.timezone.utc).timestamp()


healthy = next(epoch(line.split()[1]) for line in progress if line.startswith("healthy"))
windows = []
start = healthy
for line in progress:
    match = re.match(r"cell (\S+) rc \d+ (\S+)", line)
    if match:
        end = epoch(match.group(2))
        windows.append((match.group(1), start, end))
        start = end

steps = []
plain = []
for line in (arm / "serve.log").read_text(errors="replace").splitlines():
    match = re.search(r"graph\.plain t=(\S+) rows=(\d+)", line)
    if match:
        plain.append((float(match.group(1)), int(match.group(2))))
        continue
    match = re.search(r"graph\.(whole|prefix) t=(\S+) (.*)", line)
    if not match:
        continue
    fields = dict(item.split("=") for item in match.group(3).split())
    step = {
        "kind": match.group(1),
        "t": float(match.group(2)),
        "hit": int(fields["hit"]),
        "top": int(fields["top"]),
        "lengths": [int(v) for v in fields["lengths"].split(",")],
        "first": [int(v) for v in fields["first"].split(",")] if "first" in fields else None,
    }
    steps.append(step)


def groups(lengths: list[int], top: int) -> int:
    count, frames, rows = 1, 0, 0
    for length in lengths:
        if frames + length > top or rows == row_slots:
            count, frames, rows = count + 1, 0, 0
        frames, rows = frames + length, rows + 1
    return count


for cell, begin, end in windows:
    for kind in ("whole", "prefix"):
        cell_steps = [s for s in steps if s["kind"] == kind and begin <= s["t"] <= end]
        if not cell_steps:
            continue
        misses = [s for s in cell_steps if not s["hit"]]
        rows = sum(len(s["lengths"]) for s in cell_steps)
        hit_rows = sum(len(s["lengths"]) for s in cell_steps if s["hit"])
        top = cell_steps[0]["top"]
        oversized = [length for s in cell_steps for length in s["lengths"] if length > top]
        split = [groups(s["lengths"], top) for s in misses]
        line = (
            f"{cell} {kind}: top={top} steps={len(cell_steps)} step_hits={len(cell_steps) - len(misses)} "
            f"({100 * (len(cell_steps) - len(misses)) / len(cell_steps):.1f} %) rows={rows} "
            f"row_hits={hit_rows} ({100 * hit_rows / rows:.1f} %) misses={len(misses)} "
            f"miss_groups={sorted(set(split))} rows_over_top={len(oversized)}"
        )
        if kind == "prefix":
            later = [length for s in cell_steps for length, f in zip(s["lengths"], s["first"]) if not f]
            firsts = [length for s in cell_steps for length, f in zip(s["lengths"], s["first"]) if f]
            miss_first = [s for s in misses if any(s["first"])]
            first_share = sum(
                length for s in misses for length, f in zip(s["lengths"], s["first"]) if f
            ) / max(sum(sum(s["lengths"]) for s in misses), 1)
            plain_rows = sum(r for t, r in plain if begin <= t <= end)
            line += (
                f" | later_hop max={max(later, default=0)} first_hop max={max(firsts, default=0)} "
                f"mean={sum(firsts) / max(len(firsts), 1):.0f} misses_with_first={len(miss_first)} "
                f"first_share_of_miss_frames={100 * first_share:.1f} % "
                f"plain_hop_rows={plain_rows} ({100 * plain_rows / max(plain_rows + rows, 1):.1f} % of hop rows)"
            )
        print(line)
