"""Graph hit rate per benchmark cell from a graph_hits probe boot (ab_boot.sh arm directory):
steps, hits, rows, frames per CFG half and the tier fill of the hits, for the whole history and
the prefix solves. Lines before the server was healthy (startup warmups) are left out.
usage: python3 graph_hits_summary.py <arm dir>"""

import datetime
import re
import statistics
import sys
from pathlib import Path

arm = Path(sys.argv[1])
progress = (arm / "progress.txt").read_text().splitlines()
day = datetime.datetime.fromtimestamp(arm.joinpath("progress.txt").stat().st_mtime, datetime.timezone.utc).date()


def epoch(clock: str) -> float:
    moment = datetime.datetime.combine(day, datetime.time.fromisoformat(clock), datetime.timezone.utc)
    return moment.timestamp()


healthy = next(epoch(line.split()[1]) for line in progress if line.startswith("healthy"))
windows = []
start = healthy
for line in progress:
    match = re.match(r"cell (\S+) rc \d+ (\S+)", line)
    if match:
        end = epoch(match.group(2))
        windows.append((match.group(1), start, end))
        start = end

records = []
for line in (arm / "serve.log").read_text(errors="replace").splitlines():
    match = re.search(r"graph\.(whole|prefix) t=(\S+) (.*)", line)
    if match:
        fields = dict(item.split("=") for item in match.group(3).split())
        records.append((match.group(1), float(match.group(2)), {k: int(v) for k, v in fields.items()}))


def percentile(values: list[int], share: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


for cell, begin, end in windows:
    for kind in ("whole", "prefix"):
        rows = [f for k, t, f in records if k == kind and begin <= t <= end]
        if not rows:
            print(f"{cell} {kind}: no steps")
            continue
        hits = [f for f in rows if f["hit"]]
        misses = [f for f in rows if not f["hit"]]
        frames = [f["frames"] for f in rows]
        fill = [f["frames"] / f["tier"] for f in hits]
        print(
            f"{cell} {kind}: steps={len(rows)} hits={len(hits)} "
            f"hit_rate={100 * len(hits) / len(rows):.1f} % "
            f"rows mean={statistics.mean(f['rows'] for f in rows):.1f} max={max(f['rows'] for f in rows)} "
            f"frames mean={statistics.mean(frames):.0f} p50={percentile(frames, 0.5)} "
            f"p95={percentile(frames, 0.95)} max={max(frames)} "
            f"tier_fill mean={100 * statistics.mean(fill):.1f} % "
            f"missed_frames={sorted(f['frames'] for f in misses)[:10]}"
        )
