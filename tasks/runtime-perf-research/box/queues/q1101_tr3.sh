#!/usr/bin/env bash
# 11-01: the remaining span traces as single boots without GPU context switch collection (its nsys start can time out
# and stall the server on this host even alone). Colocated marks+3p, marks+3s (c64 again), then the two-card speech
# default with marks. c16 and c64, warm then an 8 s window.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
MARKS_ENV="SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1"
NSYS="--nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-delay 20 --nsys-seconds 8"
boot() {
  local name=$1 tree=$2 model=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model $model --card 0 --port 23000 --out $O/$name \
    --loads 16,64 --passes warm,nsys --nsys-session $name $NSYS --serve-env "$MARKS_ENV" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1101_tr3 start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot t3_p1 /workspace/marks_3p omni_colocated
boot t3_s1 /workspace/marks_3s omni_colocated
boot t3_sd /workspace/marks omni_speech
log "q1101_tr3 done"
