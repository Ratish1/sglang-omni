"""A thread's py-spy samples (collapsed format) by leaf line and by the deepest sglang_omni frame.
usage: python3 - PYSPY.txt THREAD_NAME [RATE] [SECONDS] < leaf_lines.py"""

import sys
from collections import Counter

path, thread = sys.argv[1], sys.argv[2]
rate = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
leaves, owners, total = Counter(), Counter(), 0
for line in open(path, encoding="utf-8"):
    stack, _, count = line.rstrip("\n").rpartition(" ")
    frames = stack.split(";")
    if thread not in frames[0]:
        continue
    count = int(count)
    total += count
    leaves[frames[-1]] += count
    ours = [frame for frame in frames[1:] if "sglang_omni/" in frame]
    owners[(ours[-1] if ours else "-") + "  <-  " + frames[-1]] += count
ticks = rate * seconds
print(f"{thread}: {total} samples, busy {100 * total / ticks:.0f} % of one core")
print("leaf lines:")
for frame, count in leaves.most_common(18):
    print(f"  {100 * count / ticks:5.1f} %  {frame}")
print("deepest sglang_omni frame <- leaf:")
for frame, count in owners.most_common(22):
    print(f"  {100 * count / ticks:5.1f} %  {frame}")
