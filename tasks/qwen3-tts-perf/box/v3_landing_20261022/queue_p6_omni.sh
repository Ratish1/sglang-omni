#!/usr/bin/env bash
# After the V3 landing queue: P6's Qwen3-Omni side after its merge. Main just before #3033 (4f4cf02cc) against the
# merge (dc8a48e11), Qwen3-Omni's colocated bf16 speech server on one card each, Omni SeedTTS EN full corpus at c16,
# both card orders, WER and SIM.
set -u
export PAIR_ROOT=/data/v3_landing_20261022/pairs
D=/data/v3_landing_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
BEFORE=/workspace/main_4f4cf02cc
AFTER=/workspace/main_dc8a48e11
L=$D/queue.log
until grep -q queue_v3_landing_done $L; do sleep 30; done
echo "queue p6 omni start $(date +%T) before $(git -C $BEFORE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')after $(git -C $AFTER rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')t_marks $(git -C $T rev-parse HEAD)" >> $L
run_omni_pair() {
  python3 $IN/omni_pair.py --label $1 --tree $BEFORE --b-tree $AFTER --a-card $2 --b-card $3 --port $4 > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
}
run_omni_pair p6omni_c16_a 2 3 11000
run_omni_pair p6omni_c16_b 3 2 24000
echo "queue_p6_omni_done $(date +%T)" >> $L
