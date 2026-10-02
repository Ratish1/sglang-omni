"""Kernel time of one process stage in two nsys censuses, by kernel name, largest changes first.

The stage is found by its 'proc stage=' NVTX mark. Durations are summed over the whole
session; counts tell a new or vanished kernel from a slower one.

usage: python nsys_kernel_diff.py STAGE A.sqlite B.sqlite [--top 25]
"""

import argparse
import sqlite3
from collections import defaultdict


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def kernels_of_stage(path: str, stage: str) -> dict[str, list[float]]:
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    pids = set()
    for global_tid, text, text_id in db.execute(
        "select globalTid, text, textId from NVTX_EVENTS where end is null"
    ):
        label = text if text is not None else strings.get(text_id, "")
        if label == f"proc stage={stage}":
            pids.add(pid_of(global_tid))
        else:
            pass
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for start, end, global_pid, name_id in db.execute(
        "select start, end, globalPid, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        if pid_of(global_pid) in pids:
            row = totals[strings.get(name_id, str(name_id))]
            row[0] += (end - start) / 1e6
            row[1] += 1
        else:
            pass
    return totals


parser = argparse.ArgumentParser()
parser.add_argument("stage")
parser.add_argument("first")
parser.add_argument("second")
parser.add_argument("--top", type=int, default=25)
parser.add_argument(
    "--match",
    default="",
    help="keep kernels whose name contains any of these comma separated words",
)
args = parser.parse_args()
first = kernels_of_stage(args.first, args.stage)
second = kernels_of_stage(args.second, args.stage)
names = set(first) | set(second)
words = [word for word in args.match.split(",") if word]
names = {name for name in names if not words or any(word in name for word in words)}
rows = sorted(
    names,
    key=lambda name: -abs(second.get(name, [0.0, 0])[0] - first.get(name, [0.0, 0])[0]),
)
print(
    f"total ms {sum(v[0] for v in first.values()):.1f} / {sum(v[0] for v in second.values()):.1f}"
)
print(f"{'ms first':>10} {'n first':>8} {'ms second':>10} {'n second':>8}  kernel")
for name in rows[: args.top]:
    a = first.get(name, [0.0, 0])
    b = second.get(name, [0.0, 0])
    print(f"{a[0]:10.1f} {a[1]:8d} {b[0]:10.1f} {b[1]:8d}  {name[:160]}")
