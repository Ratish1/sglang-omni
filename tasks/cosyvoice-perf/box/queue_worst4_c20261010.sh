#!/usr/bin/env bash
# main's arm of the 32 row worst step again (w32b's main arm met the py-spy server on card 0), on
# card 0 once queue_worst3_c20261010.sh is done: 32 identical long requests at 32 vocoder rows.
# usage (in the container): bash queue_worst4_c20261010.sh
R=/data/c20261010
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
PROBE=$KIT/prefix_count
until grep -q "queue done" $R/queue_worst3.txt 2> /dev/null; do sleep 20; done
echo "w32c $(date +%T)" >> $R/queue_worst4.txt
EXTRA_ENV="PYTHONPATH=/workspace/sglang-omni:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=$R/cores/core_%h_%p_%t" \
  CELLS="W32:32:32:1234:1:$R/worst32/meta.lst" SERVE_ARGS="--vocoder.factory.max_batch_size 32" \
  bash $KIT/ab_boot.sh /workspace/sglang-omni $R/w32c/card0_main 0 34000 0-15,64-79 $R/inductor_sglang-omni
echo "queue done $(date +%T)" >> $R/queue_worst4.txt
