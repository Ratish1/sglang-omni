#!/usr/bin/env bash
# After queue_aa_c16_1025.sh: D8's L1 c16 speaker similarity (against each sample's own clip) read -1.19 and -1.91
# with D8 on card 3 and +0.45 and -0.19 on card 2 over versions 1 and 3 (the same long-path code), pooled about -0.7
# with an interval across zero, while both of D8's long-reference inputs measured as close to the model's fp32 output
# as main's. Main against main on L1 c16 in both card orders, scored as D8's pairs were, gives that metric's band on
# this node.
set -u
D=/data/next1025
export PAIR_ROOT=$D/pairs
K=${T:?the kit tree}/tasks/qwen3-tts-perf/box
M=${M:?the main tree}
LONG=$D/long
SHORT=$D/short_reference_meta
L=$D/queue.log
until grep -q queue_aa_c16_1025_done $L; do sleep 30; done
echo "queue aa L1 1025 start $(date +%T)" >> $L
for spec in "aa_L1_c16_a 2 3 20600" "aa_L1_c16_b 3 2 21800"; do
  set -- $spec
  python3 $K/v3_20261020/pair.py --label $1 --tree $M --b-tree $M --concurrency 16 --a-card $2 --b-card $3 --port $4 \
    --meta $LONG/L1_c16_run.lst > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
  python3 $K/chain_20261022/similarity_excluding_caps.py $1 --tree $M --card 2 --meta $SHORT/L1_c16_run.lst > $D/similarity_$1.out 2>&1
  echo "similarity excluding caps $1 rc $? $(head -n 1 $D/similarity_$1.out) $(date +%T)" >> $L
done
python3 $K/v3_20261020/pair_table.py aa_L1_c16_a aa_L1_c16_b > $D/table_aa_L1.txt 2>&1
echo "queue_aa_l1_1025_done $(date +%T)" >> $L
