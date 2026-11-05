#!/usr/bin/env bash
# 11-06, node-0000, two H200 on separate sockets: first boots on upstream main 824d64358 (SGLang 0.5.23). The h200 and h100_bf16 colocated
# configs each compile on their card's socket; their numbers are not read. The CPU sampler stays on for every later boot.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
nohup setsid python3 $K/cpu_load_sampler.py $O/cpu_load.log > /dev/null 2>&1 < /dev/null &
fill() {
  local name=$1 model=$2 card=$3 cpus=$4 port=$5
  log "fill $name $model card $card start"
  taskset -c $cpus python3 $K/baseline.py --tree $MAIN --model $model --card $card --port $port --out $O/$name --loads 16 --passes warm > $O/$name.driver.log 2>&1
  log "fill $name rc $?"
}
log "q1106_fill start"
fill fill_a omni_colocated 0 $CPUS0 28400 & fill fill_b omni_colocated_h100_config 1 $CPUS1 28800 & wait
log "q1106_fill done"
