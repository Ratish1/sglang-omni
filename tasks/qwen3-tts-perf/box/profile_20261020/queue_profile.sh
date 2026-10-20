#!/usr/bin/env bash
# 10-20, Qwen3-TTS Base profile on main b658fa671 (PLAN.md): the clean bench, then the formal and the mapping
# captures of W1 to W4, one boot at a time on card 0.
set -u
D=/data/prof_20261020
T=/workspace/sglang-omni
IN=${IN:-/workspace/t_marks/tasks/qwen3-tts-perf/box/profile_20261020}
L=$D/queue.log
mkdir -p $D
echo "queue start $(date +%T) marks $(git -C $IN rev-parse HEAD) tree $(git -C $T rev-parse HEAD:sglang_omni)" >> $L
for mode in clean formal mapping; do
  python3 $IN/profile_cells.py --tree $T --card 0 --port 31000 --mode $mode --out $D/$mode > $D/$mode.out 2>&1
  echo "$mode rc $? $(date +%T)" >> $L
done
echo queue_done $(date +%T) >> $L
