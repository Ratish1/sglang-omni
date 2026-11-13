#!/usr/bin/env bash
# 11-13, node-0000, one H200: both final heads stacked (#3452 default-wait lane 0b7b20957 merged with #3451 baf4a97dc) against
# upstream main 35890f10c, colocated SeedTTS c16 (3 passes) and c64, boots m x x m each, before either PR branch is replaced.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main STACK=/workspace/stack
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 load $(tail -1 $O/cpu_load.log) start"; }
nohup setsid python3 $K/cpu_load_sampler.py $O/cpu_load.log > /dev/null 2>&1 < /dev/null &
sampler=$!
log "q1113_stack start"
boot() {
  local name=$1 tree=$2 load=$3 passes=$4
  note $name $tree "c$load"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads $load --passes $passes --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
boot k16_m1 $MAIN 16 warm,clean,events
boot k16_x1 $STACK 16 warm,clean,events
boot k16_x2 $STACK 16 warm,clean,events
boot k16_m2 $MAIN 16 warm,clean,events
log "cell k16 done"
boot k64_m1 $MAIN 64 warm,events
boot k64_x1 $STACK 64 warm,events
boot k64_x2 $STACK 64 warm,events
boot k64_m2 $MAIN 64 warm,events
kill $sampler
log "q1113_stack done"
