#!/usr/bin/env bash
# 11-06, node-0000, after q1106_fused. #3451 as it stands read +2.34 ms (se 1.00) on h200 unseeded, all pairs up, as on 0.5.22, so the seeded h100_bf16 read does not generalize.
# The candidate against main on the same cells, and the rest of the config and seed cross. Then #3452 on Video-AMME, the CI's FP8 colocated stage, the 0.62 floor, at the
# CI's load per replica (c16 over 2 replicas) and at c16 on one.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main P=/workspace/p3 A=/workspace/p3a L=/workspace/lane
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_fused done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 card=$3 model=$4
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $model card $card load $(tail -1 $O/cpu_load.log) start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model $model --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pairs() {
  local tag=$1 second=$2 model=$3
  local index=1
  while [ $index -le 4 ]; do
    if [ $((index % 2)) = 1 ]; then
      boot ${tag}_m$index $MAIN 0 $model & boot ${tag}_p$index $second 1 $model & wait
    else
      boot ${tag}_p$index $second 0 $model & boot ${tag}_m$index $MAIN 1 $model & wait
    fi
    index=$((index + 1))
  done
  log "cell $tag done"
}
amme() {
  local name=$1 tree=$2 card=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) video amme fp8 card $card load $(tail -1 $O/cpu_load.log) start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_fp8_colocated --card $card --port $(port $card) --out $O/$name \
    --bench videoamme --loads 8,16 --passes clean --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1106_cross2 start"
pairs ah $A omni_colocated
pairs au $A omni_colocated_h100_config
pairs pu $P omni_colocated_h100_config
amme am_m1 $MAIN 0 & amme am_l1 $L 1 & wait
amme am_l2 $L 0 & amme am_m2 $MAIN 1 & wait
log "q1106_cross2 done"
