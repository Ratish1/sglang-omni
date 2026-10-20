#!/usr/bin/env bash
# Item 5 at stream c64 with the count probe on both arms, both card orders: where the requests the AR
# no longer holds wait (the vocoder's backlog per step).
cd /data/p20261008
P=/data/p20261008
PROBE=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/prefix_count
echo "abar3 $(date +%T)" >> $P/queue.txt
CELLS="s64:64:::1" SCORE_CELLS=none \
  EXTRA_ENV_0="PYTHONPATH=/workspace/mixed:$PROBE OMNI_PREFIX_COUNT=1" \
  EXTRA_ENV_1="PYTHONPATH=/workspace/arpre:$PROBE OMNI_PREFIX_COUNT=1" \
  bash ab_pair.sh $P/abar3 mixed arpre 23000
echo "abar4 $(date +%T)" >> $P/queue.txt
CELLS="s64:64:::1" SCORE_CELLS=none \
  EXTRA_ENV_0="PYTHONPATH=/workspace/arpre:$PROBE OMNI_PREFIX_COUNT=1" \
  EXTRA_ENV_1="PYTHONPATH=/workspace/mixed:$PROBE OMNI_PREFIX_COUNT=1" \
  bash ab_pair.sh $P/abar4 arpre mixed 24000
echo "queue7 done $(date +%T)" >> $P/queue.txt
