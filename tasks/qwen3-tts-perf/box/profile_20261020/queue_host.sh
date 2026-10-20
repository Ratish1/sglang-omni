#!/usr/bin/env bash
# 10-20, item 1's host attribution (RESULTS.md 8 row 1): step windows of the engine's forwards with stacks, graphs on,
# from the trace marks tree, card 0; W2 (bs 1 decode), W3 (c16), W4 (c64).
set -u
D=/data/prof_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/profile_20261020
L=$D/queue.log
echo "queue host start $(date +%T) marks $(git -C $T rev-parse HEAD) tree $(git -C $T rev-parse HEAD:sglang_omni)" >> $L
python3 $IN/profile_cells.py --tree $T --card 0 --port 31000 --mode host --out $D/host --cells W2 W3 W4 > $D/host.out 2>&1
echo "host rc $? $(date +%T)" >> $L
echo queue_host_done $(date +%T) >> $L
