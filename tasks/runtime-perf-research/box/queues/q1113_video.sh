#!/usr/bin/env bash
# 11-13, node-0000, after q1113_stack: #3452's final form (default-wait lane 0b7b20957) against main on text Video-MME, the CI sample,
# colocated c16, boots m w w m, two clean repeats and events, for the PR's video row on this base.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main WAIT=/workspace/lane_wait
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1113_stack done" $O/queue.log 2>/dev/null; do sleep 15; done
log "q1113_video start"
video() {
  local name=$1 tree=$2
  note $name $tree "colocated text video"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video vm_m1 $MAIN
video vm_w1 $WAIT
video vm_w2 $WAIT
video vm_m2 $MAIN
log "q1113_video done"
