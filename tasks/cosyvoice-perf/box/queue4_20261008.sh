#!/usr/bin/env bash
# After queue3_20261008.sh: item 4 against item 4 with one slack ranking for started and unstarted
# streams (experiment), both card orders, count probe on both arms.
# usage (in the container): bash queue4_20261008.sh
cd /data/p20261008
P=/data/p20261008
PROBE=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/prefix_count
until grep -q "queue3 done" $P/queue.txt 2>/dev/null; do sleep 10; done
RANK_CELLS="c16:16:::1 s32:32:::1 s64:64:::1"
echo "abrank1 $(date +%T)" >> $P/queue.txt
CELLS=$RANK_CELLS SCORE_CELLS=none \
  EXTRA_ENV_0="PYTHONPATH=/workspace/mixed:$PROBE OMNI_PREFIX_COUNT=1" \
  EXTRA_ENV_1="PYTHONPATH=/workspace/rank:$PROBE OMNI_PREFIX_COUNT=1" \
  bash ab_pair.sh $P/abrank1 mixed rank 29000
echo "abrank2 $(date +%T)" >> $P/queue.txt
CELLS=$RANK_CELLS SCORE_CELLS=none \
  EXTRA_ENV_0="PYTHONPATH=/workspace/rank:$PROBE OMNI_PREFIX_COUNT=1" \
  EXTRA_ENV_1="PYTHONPATH=/workspace/mixed:$PROBE OMNI_PREFIX_COUNT=1" \
  bash ab_pair.sh $P/abrank2 rank mixed 30000
echo "queue4 done $(date +%T)" >> $P/queue.txt
