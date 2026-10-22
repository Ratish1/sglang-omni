#!/usr/bin/env bash
# After the Qwen3-Omni check: does P6 raise the frame-cap rate of the prompts that run away? Main just before #3033
# (4f4cf02cc) against the merge (dc8a48e11), Qwen3-TTS 1.7B Base, the six SeedTTS EN prompts that have ever run to the
# 2,048-frame cap, repeated (300 requests a run, the known prompt 100 of them), streaming, max concurrency 16,
# unseeded, both card orders; caps counted per prompt, no scoring.
set -u
export PAIR_ROOT=/data/v3_landing_20261022/pairs
D=/data/v3_landing_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
BEFORE=/workspace/main_4f4cf02cc
AFTER=/workspace/main_dc8a48e11
L=$D/queue.log
until grep -q queue_p6_omni_done $L; do sleep 30; done
echo "queue runaway start $(date +%T) before $(git -C $BEFORE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')after $(git -C $AFTER rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')t_marks $(git -C $T rev-parse HEAD)" >> $L
(cd $AFTER && PYTHONPATH=$AFTER python3 $HERE/runaway_meta.py $D/runaway >> $L 2>&1)
run_pair() {
  python3 $IN/pair.py --label $1 --tree $BEFORE --b-tree $AFTER --concurrency 16 --a-card $2 --b-card $3 --port $4 \
    --meta $D/runaway/meta.lst --skip-score > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
}
run_pair runaway_c16_a 2 3 13000
run_pair runaway_c16_b 3 2 18500
python3 $HERE/cap_count.py runaway_c16_a runaway_c16_b > $D/caps_runaway.txt 2>&1
echo "queue_runaway_done $(date +%T)" >> $L
