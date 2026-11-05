#!/usr/bin/env bash
# 11-06, node-0000, after q1106_stage: #3451 on upstream main 824d64358 (SGLang 0.5.23), same time, each server on its card's socket, every arm
# on both cards. Three arms against main on the cell where the cost was largest (h100_bf16, seeded): the PR as it stands (forward, sampler and
# predictor back to back, wait at the end), the candidate (wait before the predictor, a gap on the card), and the bisect arm (tokens staged
# before the predictor with no early wait, the same host order with no gap). The bisect arm decides whether the gap is the mechanism.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main
SEED="--seed 3451 --warmup 16"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_stage done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 card=$3 model=$4 bench=${5:-}
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $model [$bench] card $card load $(tail -1 $O/cpu_load.log) start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model $model --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 "--bench-extra=$bench" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pairs() {
  local tag=$1 arm=$2 count=$3 model=$4 bench=${5:-}
  local index=1
  while [ $index -le $count ]; do
    if [ $((index % 2)) = 1 ]; then
      boot ${tag}_m$index $MAIN 0 $model "$bench" & boot ${tag}_p$index $arm 1 $model "$bench" & wait
    else
      boot ${tag}_p$index $arm 0 $model "$bench" & boot ${tag}_m$index $MAIN 1 $model "$bench" & wait
    fi
    index=$((index + 1))
  done
  log "cell $tag done"
}
log "q1106_3451 start"
pairs bt /workspace/p3s 4 omni_colocated_h100_config "$SEED"
pairs ba /workspace/p3a 4 omni_colocated_h100_config "$SEED"
pairs bp /workspace/p3 2 omni_colocated_h100_config "$SEED"
log "q1106_3451 done"
