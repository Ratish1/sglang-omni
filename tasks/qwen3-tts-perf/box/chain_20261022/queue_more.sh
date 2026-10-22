#!/usr/bin/env bash
# After queue_followup.sh: D1's quality with the codes held equal, a seeded c1 pair of the chain before D1 against D1
# on the L1 c1 list (88 s references); then the cap rate on the third newly capped prompt, repeated 100 times, main
# against the chain head at c16 in both card orders.
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
CAPS=$SEEDED/cap_count.py
LONG=/data/prof_20261022/formal/meta_w2/long
M=/workspace/main_1b726ab64
C=${C:?the chain head tree}
PRE_D1=${PRE_D1:?the chain before D1}
D1=${D1:?the chain at D1}
L=$D/queue.log
until grep -q queue_followup_done $L; do sleep 20; done
echo "queue more start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
python3 $IN/pair.py --label fins_L1_c1 --tree $PRE_D1 --b-tree $D1 --concurrency 1 --a-card 2 --b-card 3 --port 7000 \
  --meta $LONG/L1_c1_run.lst --seed 1234 > $D/fins_L1_c1.out 2>&1
echo "fins_L1_c1 rc $? $(date +%T) $(grep 'byte identical' $D/fins_L1_c1.out)" >> $L
python3 $SEEDED/seeded_quality.py fins_L1_c1 > $D/seeded_quality_d1_c1.txt 2>&1
(cd $C && PYTHONPATH=$C python3 $HERE/repeat_meta.py $D/runaway3 \
  common_voice_en_25662916-common_voice_en_25663069:100 >> $L 2>&1)
for order in "a 2 3 7600" "b 3 2 8200"; do
  set -- $order
  python3 $IN/pair.py --label runaway3_c16_$1 --tree $M --b-tree $C --concurrency 16 --a-card $2 --b-card $3 --port $4 \
    --meta $D/runaway3/meta.lst --skip-score > $D/runaway3_c16_$1.out 2>&1
  echo "runaway3_c16_$1 rc $? $(date +%T)" >> $L
done
python3 $CAPS runaway3_c16_a runaway3_c16_b > $D/caps_runaway3.txt 2>&1
echo "queue_more_done $(date +%T)" >> $L
