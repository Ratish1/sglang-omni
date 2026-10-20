#!/usr/bin/env bash
# The 10-08 pairs after abtok1, one after the other on cards 0 and 1, each on its own ports below
# the ephemeral range (32768 and up), where other processes listen on random ports.
# usage (in the container): bash queue_20261008.sh
cd /data/p20261008
P=/data/p20261008
PROBE=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/prefix_count
until grep -q "pair done" $P/abtok1/progress.txt; do sleep 10; done
echo "abtok2 $(date +%T)" >> $P/queue.txt
SCORE_CELLS=none bash ab_pair.sh $P/abtok2 tokarena convk 21000
echo "aballoc1 $(date +%T)" >> $P/queue.txt
bash ab_pair.sh $P/aballoc1 tokarena alloc 22000
echo "aballoc2 $(date +%T)" >> $P/queue.txt
SCORE_CELLS=none bash ab_pair.sh $P/aballoc2 alloc tokarena 23000
POOL_CELLS="c16:16:::1 s32:32:::1 s64:64:::1 b32:32:::0 L16:16:::1:$P/longform/meta.lst"
echo "abpool1 $(date +%T)" >> $P/queue.txt
CELLS=$POOL_CELLS SCORE_CELLS=none \
  EXTRA_ENV_0="PYTHONPATH=/workspace/alloc:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1" \
  EXTRA_ENV_1="PYTHONPATH=/workspace/pool3:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1" \
  bash ab_pair.sh $P/abpool1 alloc pool3 24000
echo "abpool2 $(date +%T)" >> $P/queue.txt
CELLS=$POOL_CELLS SCORE_CELLS=none \
  EXTRA_ENV_0="PYTHONPATH=/workspace/pool3:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1" \
  EXTRA_ENV_1="PYTHONPATH=/workspace/alloc:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1" \
  bash ab_pair.sh $P/abpool2 pool3 alloc 25000
echo "queue done $(date +%T)" >> $P/queue.txt
