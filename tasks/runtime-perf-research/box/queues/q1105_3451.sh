#!/usr/bin/env bash
# 11-05, node-0000, after q1105_acc: #3451 on one node, same time, each server on its card's socket, every arm on both cards.
# Block b's h100_bf16 cost was seeded with warmup 16 and block a's h200 level was not, so config and seed are crossed here.
# Both configs use the same breakable prefill backend. Then a context switch trace of block b's setup, one session at a time.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 P=/workspace/p3 MM=/workspace/marks_main MP=/workspace/marks_3p
SEED="--seed 3451 --warmup 16"
MARKS_ENV="SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
export CUDA_DEVICE_ORDER=PCI_BUS_ID
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1105_acc done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 card=$3 model=$4 bench=${5:-}
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $model [$bench] card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model $model --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 "--bench-extra=$bench" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
cell() {
  local tag=$1 model=$2 bench=${3:-}
  boot ${tag}_m1 $MAIN 0 $model "$bench" & boot ${tag}_p1 $P 1 $model "$bench" & wait
  boot ${tag}_p2 $P 0 $model "$bench" & boot ${tag}_m2 $MAIN 1 $model "$bench" & wait
  boot ${tag}_m3 $MAIN 0 $model "$bench" & boot ${tag}_p3 $P 1 $model "$bench" & wait
  boot ${tag}_p4 $P 0 $model "$bench" & boot ${tag}_m4 $MAIN 1 $model "$bench" & wait
  log "cell $tag done"
}
trace() {
  local name=$1 tree=$2
  log "trace $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) start"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated_h100_config --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,nsys --nsys-session $name --nsys-trace cuda,nvtx,osrt --nsys-gpuctxsw --nsys-delay 15 --nsys-seconds 10 \
    --serve-env "$MARKS_ENV" "--bench-extra=$SEED" > $O/$name.driver.log 2>&1
  log "trace $name rc $?"
}
log "q1105_3451 start"
cell xbs omni_colocated_h100_config "$SEED"
cell xhs omni_colocated "$SEED"
cell xbu omni_colocated_h100_config
cell xhu omni_colocated
trace cx_m1 $MM
trace cx_p1 $MP
trace cx_p2 $MP
trace cx_m2 $MM
log "q1105_3451 done"
