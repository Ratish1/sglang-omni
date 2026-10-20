#!/usr/bin/env bash
# main's arm of the 16 row worst step again (w16b's main arm met a py-spy server on card 0 and ran
# out of memory at startup), on card 0 once queue_worst2_c20261010.sh is done.
# usage (in the container): bash queue_worst3_c20261010.sh
R=/data/c20261010
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
PROBE=$KIT/prefix_count
until grep -q "queue done" $R/queue_worst2.txt 2> /dev/null; do sleep 20; done
echo "w16c $(date +%T)" >> $R/queue_worst3.txt
EXTRA_ENV="PYTHONPATH=/workspace/sglang-omni:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=$R/cores/core_%h_%p_%t" \
  CELLS="W16:16:16:1234:1:$R/worst/meta.lst" \
  bash $KIT/ab_boot.sh /workspace/sglang-omni $R/w16c/card0_main 0 34000 0-15,64-79 $R/inductor_sglang-omni
echo "queue done $(date +%T)" >> $R/queue_worst3.txt
