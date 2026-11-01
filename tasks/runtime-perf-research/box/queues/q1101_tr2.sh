#!/usr/bin/env bash
# 11-01, after q1101_ab: span traces as single boots, one nsys session on the host at a time (two sessions starting
# GPU context switch collection at once stall both servers on this host). Colocated marks main, marks+3s, marks+3p,
# main again, on card 0; then the two-card speech default with marks. c16 and c64, warm then an 8 s window.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_ab done" $O/queue.log 2>/dev/null; do sleep 30; done
MARKS_ENV="SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1"
NSYS="--nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-gpuctxsw --nsys-delay 20 --nsys-seconds 8"
boot() {
  local name=$1 tree=$2 model=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model $model --card 0 --port 23000 --out $O/$name \
    --loads 16,64 --passes warm,nsys --nsys-session $name $NSYS --serve-env "$MARKS_ENV" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1101_tr2 start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot t2_m1 /workspace/marks omni_colocated
boot t2_s1 /workspace/marks_3s omni_colocated
boot t2_p1 /workspace/marks_3p omni_colocated
boot t2_m2 /workspace/marks omni_colocated
boot t2_sd /workspace/marks omni_speech
log "q1101_tr2 done"
