#!/usr/bin/env bash
# A/A pair (KIT_AUDIT validation 2): main 204ba29a5 (t_m) on both cards at the same time, twice, stream c16 three
# times and c64 once per pair, no scoring. Sets what a same time pair resolves: the per cell same time delta's mean
# and spread, the card effect, the first against later cells (the warm prefix and reference caches).
# usage (in the container): bash queue_aa_c20261013.sh
set -u
R=/data/c20261013
IN=$R/in
OUT=$R/aa
mkdir -p $OUT
export ROOT=$R
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
export SCORE_CELLS="none"
export CELLS="c16a:16:::1 c16b:16:::1 c16c:16:::1 c64:64:::1"
for pair in 1 2; do
  echo "aa $pair $(date +%T) card0=t_m card1=t_m" >> $OUT/queue.txt
  env $PAIR bash $IN/ab_pair_sync.sh $OUT/p$pair t_m t_m 41000
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
