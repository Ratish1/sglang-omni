#!/usr/bin/env bash
# After queue7: the vocoder's step at 32 rows against 16 on item 4's tree, both card orders.
cd /data/p20261008
P=/data/p20261008
until grep -q "queue7 done" $P/queue.txt 2>/dev/null; do sleep 10; done
CELLS8="c16:16:::1 s32:32:::1 s64:64:::1 b16:16:::0"
echo "abvb1 $(date +%T)" >> $P/queue.txt
CELLS=$CELLS8 SCORE_CELLS=none SERVE_ARGS_1="--vocoder.factory.max_batch_size 32" \
  bash ab_pair.sh $P/abvb1 mixed mixed 25000
echo "abvb2 $(date +%T)" >> $P/queue.txt
CELLS=$CELLS8 SCORE_CELLS=none SERVE_ARGS_0="--vocoder.factory.max_batch_size 32" \
  bash ab_pair.sh $P/abvb2 mixed mixed 26000
echo "queue8 done $(date +%T)" >> $P/queue.txt
