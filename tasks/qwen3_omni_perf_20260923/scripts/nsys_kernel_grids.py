"""Launch configuration and duration of kernels whose name matches, in one process stage.

Groups by (grid, block, registers, shared memory) and prints count and median duration, so a
kernel launched with a grid far larger than its work shows up as one config at a flat time.

usage: python nsys_kernel_grids.py serve.sqlite STAGE NAME_SUBSTRING [NAME_SUBSTRING ...]
"""

import sqlite3
import statistics
import sys
from collections import defaultdict


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


path, stage = sys.argv[1], sys.argv[2]
words = sys.argv[3:]
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
name_ids = {
    name_id: value
    for name_id, value in strings.items()
    if any(word in value for word in words)
}
groups = defaultdict(list)
for (
    start,
    end,
    global_pid,
    short_name,
    grid_x,
    grid_y,
    grid_z,
    block_x,
    block_y,
    block_z,
    registers,
    static_shared,
    dynamic_shared,
) in db.execute(
    "select start, end, globalPid, shortName, gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread, staticSharedMemory, dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL"
):
    if short_name in name_ids and pid_of(global_pid) in pids:
        key = (
            name_ids[short_name][:40],
            f"grid {grid_x}x{grid_y}x{grid_z}",
            f"block {block_x}x{block_y}x{block_z}",
            f"regs {registers} smem {static_shared}+{dynamic_shared}",
        )
        groups[key].append((end - start) / 1e3)
    else:
        pass
print(f"== {path} stage {stage}")
for key, durations in sorted(groups.items(), key=lambda item: -len(item[1]))[:24]:
    print(
        f"  n {len(durations):7d}  p50 {statistics.median(durations):8.2f} us  {'  '.join(key)}"
    )
