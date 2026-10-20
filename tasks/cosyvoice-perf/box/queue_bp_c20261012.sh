#!/usr/bin/env bash
# Item B (Flow step inputs without a host wait, 7f612680f on main 870d77446, tree t_b) against main (t_bm): same time
# pairs in both card orders on the barrier runner, seeded c1 and c16 with WER and SIM, plus the seeded c1 audio
# compared file by file.
# usage (in the container): bash queue_bp_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
OUT=$R/bp
mkdir -p $OUT
export ROOT=$R
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
export SCORE_CELLS="c16"
export CELLS="c1:1:64:1234:1 c16:16:::1"
echo "bp a $(date +%T) card0=t_bm card1=t_b" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/a t_bm t_b 39000
echo "bp b $(date +%T) card0=t_b card1=t_bm" >> $OUT/queue.txt
env $PAIR bash $IN/ab_pair_sync.sh $OUT/b t_b t_bm 39000
for pair in a b; do
  main_dir=$(ls -d $OUT/$pair/card*_t_bm)
  b_dir=$(ls -d $OUT/$pair/card*_t_b)
  same=0; total=0
  for wav in $main_dir/c1/audio/*; do
    total=$((total + 1))
    cmp -s "$wav" "$b_dir/c1/audio/$(basename "$wav")" && same=$((same + 1))
  done
  echo "bp $pair seeded c1 audio identical: $same of $total" >> $OUT/queue.txt
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
