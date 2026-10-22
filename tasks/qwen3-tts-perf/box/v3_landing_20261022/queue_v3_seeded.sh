#!/usr/bin/env bash
# V3's quality with the talker's codes held equal: seeded c1 pairs of main 3d7d2c306 against V3 (source tree
# 4735a6004), full SeedTTS EN, both card orders, WER and SIM; V3 changes only the vocoder, which feeds nothing back to
# the talker, so a seed gives both arms the same codes and every audio difference is V3's. After the profile queues,
# so both cards are free.
set -u
export PAIR_ROOT=/data/v3_landing_20261022/pairs
D=/data/v3_landing_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
M=/workspace/main_3d7d2c306
V3=/workspace/v3pr_4d26f2092
L=$D/queue.log
until grep -q queue_profile_fix_done /data/prof_20261022/queue.log; do sleep 20; done
echo "queue v3 seeded start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) v3 $(git -C $V3 rev-parse HEAD:sglang_omni) t_marks $(git -C $T rev-parse HEAD)" >> $L
run_pair() {
  python3 $IN/pair.py --label $1 --tree $M --b-tree $V3 --concurrency 1 --a-card $2 --b-card $3 --port $4 --seed 1234 \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) $(grep 'byte identical' $D/$1.out)" >> $L
}
run_pair v3s_c1_a 2 3 7000
run_pair v3s_c1_b 3 2 13000
python3 $HERE/seeded_quality.py v3s_c1_a v3s_c1_b > $D/seeded_quality.txt 2>&1
echo "queue_v3_seeded_done $(date +%T)" >> $L
