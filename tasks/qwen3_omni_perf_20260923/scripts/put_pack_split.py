"""shm_put_size_probe records per process and function, split by whether the pack received accelerator tensors.

Per group: count, wall p50 / p90 / p99 / max, thread CPU p50 / p90 (us) and the packed bytes p50. With a passes file
(lines "<name> start|end <epoch seconds>") the records are split by pass as well.

usage: python3 put_pack_split.py RECORDS_DIR [PASSES_FILE]
"""

import collections
import glob
import json
import os
import sys
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


windows: dict[str, dict[str, float]] = {}
if len(sys.argv) > 2:
    for line in Path(sys.argv[2]).read_text().splitlines():
        name, edge, stamp = line.split()
        windows.setdefault(name, {})[edge] = float(stamp) * 1e9
else:
    pass

groups = collections.defaultdict(list)
for path in glob.glob(os.path.join(sys.argv[1], "put_*.jsonl")):
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        window = next(
            (
                name
                for name, edges in windows.items()
                if edges["start"]
                <= record["start_ns"]
                <= edges.get("end", float("inf"))
            ),
            "all" if not windows else "outside",
        )
        key = (window, record["process"], record["function"], record["cuda_inputs"] > 0)
        groups[key].append((record["wall_us"], record["cpu_us"], record["bytes"]))

for key in sorted(groups):
    rows = groups[key]
    walls = [row[0] for row in rows]
    cpus = [row[1] for row in rows]
    print(
        f"{key[0]:7s} {key[1][:22]:22s} {key[2]:24s} device={int(key[3])} n {len(rows):5d} "
        f"wall p50 {percentile(walls, 0.5):8.1f} p90 {percentile(walls, 0.9):8.1f} "
        f"p99 {percentile(walls, 0.99):8.1f} max {max(walls):8.1f} us | cpu p50 {percentile(cpus, 0.5):7.1f} "
        f"p90 {percentile(cpus, 0.9):7.1f} | bytes p50 {percentile([row[2] for row in rows], 0.5)}"
    )
