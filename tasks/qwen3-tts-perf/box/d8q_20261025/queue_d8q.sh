#!/usr/bin/env bash
# #3177 (52983c59d; production code that of the measured b17546a65 and 0d69570a7) against its base main 67476823a on
# long references, quality only, main against the PR: the 10-25 L1 pairs read the PR's speaker similarity lower in
# both (-1.91 and -0.19) with WER level, so six more scored L1 c16 pairs (three in each card order) and four scored
# L3 c16 pairs (about 400-word targets, never scored), each scored for WER with 64 transcriptions in flight and for
# similarity against each sample's own SeedTTS clip; then long_quality.py over every pair. Cards 2 and 3.
set -u
D=/data/d8q1025
export PAIR_ROOT=$D/pairs
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
HERE=$K/chain_20261022
PROFILE=$K/profile_20261020
M=${M:?the main tree}
P=${P:?the PR tree}
LONG=$D/long
SHORT=$D/short_reference_meta
L=$D/queue.log
mkdir -p $D
echo "queue d8q start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')pr $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $M && PYTHONPATH=$M python3 $PROFILE/long_cells_meta.py $LONG > $D/long_meta.log 2>&1)
echo "long lists rc $? $(date +%T)" >> $L
(cd $M && PYTHONPATH=$M python3 $PROFILE/short_reference_meta.py $LONG $SHORT L1_c16 L3_c16 > $D/short_meta.log 2>&1)
echo "short reference lists rc $? $(wc -l < $SHORT/L1_c16_run.lst) L1 rows $(wc -l < $SHORT/L3_c16_run.lst) L3 rows $(date +%T)" >> $L
run_pair() {
  local label=$1 cell=$2
  shift 2
  python3 $K/v3_20261020/pair.py --label $label --tree $M --b-tree $P --concurrency 16 --meta $LONG/${cell}_run.lst "$@" \
    > $D/$label.out 2>&1
  local rc=$?
  python3 $HERE/similarity_excluding_caps.py $label --tree $M --card 2 --meta $SHORT/${cell}_run.lst \
    > $D/similarity_$label.out 2>&1
  echo "$label rc $rc similarity rc $? $(head -n 1 $D/similarity_$label.out) $(date +%T)" >> $L
}
run_pair q_L1_1 L1_c16 --a-card 2 --b-card 3 --port 4400
run_pair q_L1_2 L1_c16 --a-card 3 --b-card 2 --port 7000
run_pair q_L1_3 L1_c16 --a-card 2 --b-card 3 --port 8200
run_pair q_L1_4 L1_c16 --a-card 3 --b-card 2 --port 10000
run_pair q_L1_5 L1_c16 --a-card 2 --b-card 3 --port 11200
run_pair q_L1_6 L1_c16 --a-card 3 --b-card 2 --port 12400
run_pair q_L3_1 L3_c16 --a-card 2 --b-card 3 --port 13600
run_pair q_L3_2 L3_c16 --a-card 3 --b-card 2 --port 14800
run_pair q_L3_3 L3_c16 --a-card 2 --b-card 3 --port 17000
run_pair q_L3_4 L3_c16 --a-card 3 --b-card 2 --port 18200
python3 $K/d8q_20261025/long_quality.py $PAIR_ROOT/q_L1_* > $D/quality_L1.txt 2>&1
python3 $K/d8q_20261025/long_quality.py $PAIR_ROOT/q_L3_* > $D/quality_L3.txt 2>&1
python3 $K/v3_20261020/pair_table.py q_L1_1 q_L1_2 q_L1_3 q_L1_4 q_L1_5 q_L1_6 q_L3_1 q_L3_2 q_L3_3 q_L3_4 > $D/table_d8q.txt 2>&1
echo "queue_d8q_done $(date +%T)" >> $L
