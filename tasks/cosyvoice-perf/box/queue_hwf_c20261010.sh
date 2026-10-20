#!/usr/bin/env bash
# Step 1's final code (v3, t_hw3) against main (t_main) without the probe, cards 1 and 2, both
# orders: seeded stream c1, stream c16 (WER and SIM), stream c64. hwa and hwb ran v1, which v3
# changes in memory only; c1 req/s changed sign between those and the v3 probe pair.
# usage (in the container): bash queue_hwf_c20261010.sh
R=/data/c20261010h
IN=$R/in
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c64:64:::1"
export SCORE_CELLS="c16"
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"
echo "hwfa $(date +%T) card1=t_main card2=t_hw3" >> $R/queue_hwf.txt
env $PAIR bash $IN/ab_pair.sh $R/hwfa t_main t_hw3 33000
echo "hwfb $(date +%T) card1=t_hw3 card2=t_main" >> $R/queue_hwf.txt
env $PAIR bash $IN/ab_pair.sh $R/hwfb t_hw3 t_main 33000
echo "queue done $(date +%T)" >> $R/queue_hwf.txt
