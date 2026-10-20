#!/usr/bin/env bash
# Skinny GEMM candidates again (queue_bench2.sh's run stopped at a config past the shared memory limit), now with each
# Triton config also under programmatic dependent launch with its weights prefetched before the wait. Runs alone on
# card 0 between queue_v4.sh and queue_p5.sh.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
S=$T/tasks/omni_step_profiling_20260918/scripts
L=$D/queue.log
until grep -q queue_v4_done $L; do sleep 30; done
(cd /workspace/m_bcd0d8c68 && TRITON_CACHE_DIR=$D/triton_skinny timeout 2700 python3 $S/skinny_gemm_bench.py > $D/skinny_gemm_bench.txt 2>&1)
echo "skinny gemm bench rc $? $(date +%T)" >> $L
echo "queue_bench3_done $(date +%T)" >> $L
