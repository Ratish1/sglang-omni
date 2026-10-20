#!/usr/bin/env bash
# 10-20, Qwen3-TTS Base profile (PLAN.md): the formal and the mapping captures of W1 to W4 from the profiling branch
# eb2926720 (main b658fa671 plus one torch span per scheduler forward), one boot at a time on card 0. The clean bench
# ran on main in queue_profile.sh.
set -u
D=/data/prof_20261020
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/profile_20261020
L=$D/queue.log
echo "queue 2 start $(date +%T) marks $(git -C $T rev-parse HEAD) tree $(git -C $T rev-parse HEAD:sglang_omni)" >> $L
for mode in formal mapping; do
  python3 $IN/profile_cells.py --tree $T --card 0 --port 31000 --mode $mode --out $D/$mode > $D/$mode.out 2>&1
  echo "$mode rc $? $(date +%T)" >> $L
done
echo queue_done $(date +%T) >> $L
