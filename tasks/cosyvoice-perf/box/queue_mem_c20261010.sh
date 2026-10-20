#!/usr/bin/env bash
# Item 3 memory and no regression check on the ported stack (prerequisites 1 and 2, items 3 and 4)
# against main, same time, cards 1 and 2 (NUMA node 1), the count probe with its memory fields and
# GPU core dumps armed. Cells: stream c16 and c64, buffered c32, long form c16 (48 x 10 texts) and
# the length stress c16 (32 x 40 texts, finals near the AR's context). m1 at the default 16
# vocoder rows, m2 at 32 rows with the cards swapped.
# usage (in the container): bash queue_mem_c20261010.sh
R=/data/c20261010
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
PROBE=$KIT/prefix_count
CPUS1=32-47,96-111
CPUS2=48-63,112-127
mkdir -p $R/cores

arm() {
  local tree=$1 card=$2 port=$3 cpus=$4 out=$5 serve_args=${6:-}
  EXTRA_ENV="PYTHONPATH=/workspace/$tree:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=$R/cores/core_%h_%p_%t" \
    CELLS="s16:16:::1 s64:64:::1 b32:32:::0 L16:16:::1:$R/longform/meta.lst X16:16:::1:$R/stress/meta.lst" \
    SERVE_ARGS="$serve_args" \
    bash $KIT/ab_boot.sh "/workspace/$tree" "$out" "$card" "$port" "$cpus" "$R/inductor_$tree"
}

echo "m1 $(date +%T)" >> $R/queue_mem.txt
arm t_stack 1 29000 $CPUS1 $R/m1/card1_stack &
arm sglang-omni 2 29200 $CPUS2 $R/m1/card2_main &
wait
echo "m2 $(date +%T)" >> $R/queue_mem.txt
arm sglang-omni 1 29000 $CPUS1 $R/m2/card1_main "--vocoder.factory.max_batch_size 32" &
arm t_stack 2 29200 $CPUS2 $R/m2/card2_stack "--vocoder.factory.max_batch_size 32" &
wait
echo "queue done $(date +%T)" >> $R/queue_mem.txt
