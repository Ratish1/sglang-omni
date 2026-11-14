#!/usr/bin/env bash
# 11-14, node-0000, one H200 (card 0, NUMA 1). Video-AMME on the FP8 checkpoint, main against the lane.
# c1 on one fresh boot per tree: no co-batching, so the replies must match byte for byte.
# c16 on one fresh boot per tree, the CI condition (cold prompts, one pass): the answer diff names the questions batching moves.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main WAIT=/workspace/lane_wait
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
card_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  until [ "${used:-0}" -lt 1000 ]; do
    log "card busy ${used} MiB, waiting"
    sleep 30
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  done
}
log "q1114_fp8 start"
boot() {
  local name=$1 tree=$2 loads=$3
  card_free
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) fp8 videoamme c$loads start"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_fp8_colocated --card 0 --port 28400 --out $O/$name \
    --bench videoamme --loads $loads --passes clean --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
boot f8_m1 $MAIN 1
boot f8_w1 $WAIT 1
boot f8_w16 $WAIT 16
boot f8_m16 $MAIN 16
log "q1114_fp8 done"
