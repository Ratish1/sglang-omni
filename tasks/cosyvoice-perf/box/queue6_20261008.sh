#!/usr/bin/env bash
# Item 5 (the AR's prefill through SGLang's breakable CUDA graph) against item 4, both card orders.
# usage (in the container): bash queue6_20261008.sh
cd /data/p20261008
P=/data/p20261008
CELLS6="c1:1:64:1234:1 c16:16:::1 s32:32:::1 s64:64:::1 b16:16:::0"
echo "abar1 $(date +%T)" >> $P/queue.txt
CELLS=$CELLS6 SCORE_CELLS=c16 bash ab_pair.sh $P/abar1 mixed arpre 21000
echo "abar2 $(date +%T)" >> $P/queue.txt
CELLS=$CELLS6 SCORE_CELLS=none bash ab_pair.sh $P/abar2 arpre mixed 22000
echo "queue6 done $(date +%T)" >> $P/queue.txt
