#!/usr/bin/env bash
# After queue8: 32 vocoder rows with the reserve scaled to 8 GB, one boot with the memory probe on
# the cells that failed (buffered c16) and the heaviest (stream c64), card 0.
cd /data/p20261008
P=/data/p20261008
PROBE=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/prefix_count
until grep -q "queue8 done" $P/queue.txt 2>/dev/null; do sleep 10; done
echo "mem32r8 $(date +%T)" >> $P/queue.txt
CELLS="b16:16:::0 s64:64:::1" \
  EXTRA_ENV="PYTHONPATH=/workspace/mixed:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 CUDA_COREDUMP_FILE=/data/p20261008/cores/core_%h_%p_%t" \
  SERVE_ARGS="--vocoder.factory.max_batch_size 32 --vocoder.factory.flow_prefix_cache_reserve_gb 8" \
  bash /workspace/tools/tasks/cosyvoice_utilization_20260912/stage4/ab_boot.sh /workspace/mixed $P/mem32r8 0 21000 0-15,64-79 $P/inductor_mixed
echo "queue9 done $(date +%T)" >> $P/queue.txt
