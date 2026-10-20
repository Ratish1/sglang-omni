#!/usr/bin/env bash
# The vocoder's admissible worst step against item 3's reserve: 16 identical requests near the
# AR's context limit, seeded so their finals land in one step, on main (card 0) and the stack
# (card 3) at once, the count probe with its memory fields, GPU core dumps armed; w16 at the
# default 16 vocoder rows, w32 with 32 such requests at 32 rows.
# usage (in the container): bash queue_worst_c20261010.sh
R=/data/c20261010
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
PROBE=$KIT/prefix_count

arm() {
  local tree=$1 card=$2 port=$3 cpus=$4 out=$5 cells=$6 serve_args=${7:-}
  EXTRA_ENV="PYTHONPATH=/workspace/$tree:$PROBE OMNI_PREFIX_COUNT=1 OMNI_MEMORY_PROBE=1 \
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
CUDA_COREDUMP_FILE=$R/cores/core_%h_%p_%t" \
    CELLS="$cells" \
    SERVE_ARGS="$serve_args" \
    bash $KIT/ab_boot.sh "/workspace/$tree" "$out" "$card" "$port" "$cpus" "$R/inductor_$tree"
}

echo "w16 $(date +%T)" >> $R/queue_worst.txt
W16="W16:16:16:1234:1:$R/worst/meta.lst"
W32="W32:32:32:1234:1:$R/worst32/meta.lst"
arm sglang-omni 0 34000 0-15,64-79 $R/w16/card0_main "$W16" &
arm t_stack 3 34200 16-31,80-95 $R/w16/card3_stack "$W16" &
wait
echo "w32 $(date +%T)" >> $R/queue_worst.txt
arm sglang-omni 0 34000 0-15,64-79 $R/w32/card0_main "$W32" "--vocoder.factory.max_batch_size 32" &
arm t_stack 3 34200 16-31,80-95 $R/w32/card3_stack "$W32" "--vocoder.factory.max_batch_size 32" &
wait
echo "queue done $(date +%T)" >> $R/queue_worst.txt
