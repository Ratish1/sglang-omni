#!/usr/bin/env bash
# After queue_d8q.sh: the L3 pairs' transcriptions were refused (the ASR server admits 4 long-audio transcriptions at
# once, and 64 were sent), so their existing outputs are scored again with 4 in flight, then their similarity against
# each sample's own clip, then long_quality.py over the six L1 pairs and the four L3 pairs.
set -u
D=/data/d8q1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
M=${M:?the main tree}
P=${P:?the PR tree}
LONG=$D/long
SHORT=$D/short_reference_meta
L=$D/queue.log
until grep -q queue_d8q_done $L; do
  sleep 20
done
echo "queue d8q rescore start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
for order in "1 2 3 4400" "2 3 2 7000" "3 2 3 8200" "4 3 2 10000"; do
  set -- $order
  python3 $K/v3_20261020/pair.py --label q_L3_$1 --tree $M --b-tree $P --concurrency 16 --a-card $2 --b-card $3 \
    --port $4 --meta $LONG/L3_c16_run.lst --score-only --asr-concurrency 4 > $D/rescore_q_L3_$1.out 2>&1
  local_rc=$?
  python3 $HERE/similarity_excluding_caps.py q_L3_$1 --tree $M --card 2 --meta $SHORT/L3_c16_run.lst \
    > $D/similarity_q_L3_$1.out 2>&1
  echo "q_L3_$1 rescore rc $local_rc similarity rc $? $(head -n 1 $D/similarity_q_L3_$1.out | cut -c1-120) $(date +%T)" >> $L
done
python3 $K/d8q_20261025/long_quality.py $PAIR_ROOT/q_L1_* > $D/quality_L1.txt 2>&1
python3 $K/d8q_20261025/long_quality.py $PAIR_ROOT/q_L3_* > $D/quality_L3.txt 2>&1
echo "queue_d8q_rescore_done $(date +%T)" >> $L
