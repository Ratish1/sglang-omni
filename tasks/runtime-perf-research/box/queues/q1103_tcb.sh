#!/usr/bin/env bash
# 11-03, node-0001: #3451's device trace on the setup where its first audio cost shows (h100_bf16 config on the H200, seed
# 3451, warmup 16), boots m p p m, CUDA, NVTX and OSRT over 10 s of a c16 pass. tc_* on the h200 config showed main's
# idle gap between forward and sampler is the thinker's slot and 3p closes it, with the thinker's launch lag level; this
# reads whether that lag moves here. Read with extend_bursts.py and span_table.py.
set -u
O=/data/rp1103w
K=/workspace/kit_t/tasks/runtime-perf-research/box
M=/workspace/marks_main P=/workspace/marks_3p
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
export CUDA_DEVICE_ORDER=PCI_BUS_ID
MARKS_ENV="SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
boot() {
  local name=$1 tree=$2
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated_h100_config --card 0 --port 27000 --out $O/$name \
    --loads 16 --passes warm,nsys --nsys-session $name --nsys-trace cuda,nvtx,osrt --nsys-delay 15 --nsys-seconds 10 \
    --serve-env "$MARKS_ENV" "--bench-extra=--seed 3451 --warmup 16" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1103_tcb start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot tcb_m1 $M
boot tcb_p1 $P
boot tcb_p2 $P
boot tcb_m2 $M
cd $O
for b in tcb_m1 tcb_p1 tcb_p2 tcb_m2; do
  taskset -c 0-55 python3 $K/extend_bursts.py $b/c16_nsys/trace.sqlite $b/serve.log
done > $O/read_tcb.txt 2>&1
log "q1103_tcb done"
