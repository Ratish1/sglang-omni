#!/usr/bin/env bash
# The stack through item 4 (items 1 and 2, both prerequisites, items 3 and 4) against main, both card
# orders, at c1 and c16 only: the high concurrency effects of items 3 and 4 are measured in their
# own pairs.
cd /data/p20261008
P=/data/p20261008
CELLS10="c1:1:64:1234:1 c16:16:::1 b16:16:::0"
echo "abstack1 $(date +%T)" >> $P/queue.txt
CELLS=$CELLS10 SCORE_CELLS="c16 b16" bash ab_pair.sh $P/abstack1 sglang-omni mixed 29000
echo "abstack2 $(date +%T)" >> $P/queue.txt
CELLS=$CELLS10 SCORE_CELLS=none bash ab_pair.sh $P/abstack2 mixed sglang-omni 30000
echo "queue10 done $(date +%T)" >> $P/queue.txt
