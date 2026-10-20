#!/usr/bin/env bash
# The count probe (per vocoder step: plan, rows, wall ms, ready waits) on t_main against t_hw3 (step 1
# v3), cards 1 and 2, seeded stream c1 and stream c16, to find where step 1's c1 request time and c16
# inter chunk p99 go (both repeated in hwa and hwb).
# usage (in the container): bash queue_hwp_c20261010.sh
R=/data/c20261010h
IN=$R/in
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1"
export SCORE_CELLS="none"
export EXTRA_ENV_0="PYTHONPATH=/workspace/t_main:$KIT/prefix_count OMNI_PREFIX_COUNT=1"
export EXTRA_ENV_1="PYTHONPATH=/workspace/t_hw3:$KIT/prefix_count OMNI_PREFIX_COUNT=1"
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"
echo "hwp $(date +%T) card1=t_main card2=t_hw3 probe" >> $R/queue_hwp.txt
env $PAIR bash $IN/ab_pair.sh $R/hwp t_main t_hw3 33000
echo "queue done $(date +%T)" >> $R/queue_hwp.txt
