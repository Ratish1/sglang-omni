#!/usr/bin/env bash
# Skinny GEMM candidates for the talker's and predictor's dense projections at 1 to 64 rows (cuBLAS, SGLang's Hopper
# GEMV, a Triton split-K GEMM), the core shared by items P7 and T1. Runs alone on card 0 between queue_v5.sh and
# queue_v4.sh, so no pair shares the device.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
S=$T/tasks/omni_step_profiling_20260918/scripts
L=$D/queue.log
until grep -q queue_v5_done $L; do sleep 30; done
(cd /workspace/m_bcd0d8c68 && TRITON_CACHE_DIR=$D/triton_skinny timeout 2400 python3 $S/skinny_gemm_bench.py > $D/skinny_gemm_bench.txt 2>&1)
echo "skinny gemm bench rc $? $(date +%T)" >> $L
echo "queue_bench2_done $(date +%T)" >> $L
