#!/usr/bin/env bash
# The chain's cap rate against main on the two prompts that capped in fin_c16_b's chain arm: each repeated 100 times,
# streaming, c16, unseeded, both card orders, no scoring; after queue_final.sh.
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
CAPS=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022/cap_count.py
M=/workspace/main_1b726ab64
C=${C:?the chain head tree}
L=$D/queue.log
until grep -q queue_final_done $L; do sleep 20; done
echo "queue runaway start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) chain $(git -C $C rev-parse HEAD:sglang_omni) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $C && PYTHONPATH=$C python3 $HERE/repeat_meta.py $D/runaway \
  common_voice_en_20798109-common_voice_en_20798113:100 common_voice_en_19916471-common_voice_en_19916474:100 >> $L 2>&1)
run_pair() {
  python3 $IN/pair.py --label $1 --tree $M --b-tree $C --concurrency 16 --a-card $2 --b-card $3 --port $4 \
    --meta $D/runaway/meta.lst --skip-score > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
}
run_pair runaway_c16_a 2 3 15400
run_pair runaway_c16_b 3 2 16600
python3 $CAPS runaway_c16_a runaway_c16_b > $D/caps_runaway.txt 2>&1
echo "queue_runaway_done $(date +%T)" >> $L
