#!/usr/bin/env bash
# Item 3's reserve sized for the admissible worst step, as a knob test before any rule change: the
# stack at 32 vocoder rows with a 12 GB reserve on the 32 request worst step (card 3), and at 16
# rows with a 6 GB reserve on the 16 request worst step, stream c64 and long form c16 (card 0).
# usage (in the container): bash queue_reserve_c20261010.sh
R=/data/c20261010
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
PROBE=$KIT/prefix_count

arm() {
  local card=$1 port=$2 cpus=$3 out=$4 cells=$5 serve_args=$6
  EXTRA_ENV="PYTHONPATH=/workspace/t_stack:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=$R/cores/core_%h_%p_%t" \
    CELLS="$cells" SERVE_ARGS="$serve_args" \
    bash $KIT/ab_boot.sh /workspace/t_stack "$out" "$card" "$port" "$cpus" "$R/inductor_t_stack"
}

echo "r $(date +%T)" >> $R/queue_reserve.txt
arm 3 34200 16-31,80-95 $R/r32/card3_stack "W32:32:32:1234:1:$R/worst32/meta.lst s64:64:::1" \
  "--vocoder.factory.max_batch_size 32 --vocoder.factory.flow_prefix_cache_reserve_gb 12" &
arm 0 34000 0-15,64-79 $R/r16/card0_stack \
  "W16:16:16:1234:1:$R/worst/meta.lst s64:64:::1 L16:16:::1:$R/longform/meta.lst" \
  "--vocoder.factory.flow_prefix_cache_reserve_gb 6" &
wait
echo "queue done $(date +%T)" >> $R/queue_reserve.txt
