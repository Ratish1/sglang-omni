#!/usr/bin/env bash
# Two more V3 c16 pairs against main 3d7d2c306, one in each card order: both c16 pairs of queue_v3_landing.sh had an
# arm run a prompt to the frame cap, and the PR's table reads req/s from a cap-free pair.
set -u
export PAIR_ROOT=/data/v3_landing_20261022/pairs
D=/data/v3_landing_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
M=/workspace/main_3d7d2c306
V3=/workspace/v3pr_4d26f2092
L=$D/queue.log
echo "queue v3 c16 more start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) v3 $(git -C $V3 rev-parse HEAD:sglang_omni) t_marks $(git -C $T rev-parse HEAD)" >> $L
run_pair() {
  python3 $IN/pair.py --label $1 --tree $M --b-tree $V3 --concurrency 16 --a-card $2 --b-card $3 --port $4 --gpu-metrics \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)" >> $L
}
run_pair v3m4_c16_c 2 3 20000
run_pair v3m4_c16_d 3 2 27000
labels="v3m4_c16_c v3m4_c16_d"
python3 $IN/pair_table.py $labels > $D/table_v3m4_more.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_v3m4_more.txt 2>&1
python3 $READ/first_audio_tail.py $labels > $D/first_audio_tail_v3m4_more.txt 2>&1
echo "queue_v3_c16_more_done $(date +%T)" >> $L
