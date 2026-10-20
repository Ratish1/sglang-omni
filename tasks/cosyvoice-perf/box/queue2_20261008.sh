#!/usr/bin/env bash
# After queue_20261008.sh: item 4 (one step for both decode kinds) against item 3 in both card
# orders with the count probe, then the GPU unit tests of the whole chain.
# usage (in the container): bash queue2_20261008.sh
cd /data/p20261008
P=/data/p20261008
PROBE=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/prefix_count
until grep -q "queue done" $P/queue.txt 2>/dev/null; do sleep 10; done
MIXED_CELLS="c1:1:64:1234:1 c16:16:::1 s32:32:::1 s64:64:::1"
echo "abmixed1 $(date +%T)" >> $P/queue.txt
CELLS=$MIXED_CELLS SCORE_CELLS=c16 \
  EXTRA_ENV_0="PYTHONPATH=/workspace/pool3:$PROBE OMNI_PREFIX_COUNT=1" \
  EXTRA_ENV_1="PYTHONPATH=/workspace/mixed:$PROBE OMNI_PREFIX_COUNT=1" \
  bash ab_pair.sh $P/abmixed1 pool3 mixed 26000
echo "abmixed2 $(date +%T)" >> $P/queue.txt
CELLS=$MIXED_CELLS SCORE_CELLS=none \
  EXTRA_ENV_0="PYTHONPATH=/workspace/mixed:$PROBE OMNI_PREFIX_COUNT=1" \
  EXTRA_ENV_1="PYTHONPATH=/workspace/pool3:$PROBE OMNI_PREFIX_COUNT=1" \
  bash ab_pair.sh $P/abmixed2 mixed pool3 27000
echo "unit mixed $(date +%T)" >> $P/queue.txt
bash $P/unit/run_unit.sh mixed 0 $P/unit/mixed_gpu.log
echo "queue2 done $(date +%T)" >> $P/queue.txt
