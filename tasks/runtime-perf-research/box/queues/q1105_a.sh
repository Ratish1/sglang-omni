#!/usr/bin/env bash
# 11-05, node-0000, after q1105_fp: #3451 with its one token wait before the predictor (b6309c620), against main, same time, each
# server on its card's socket. 3p closes the idle gap between forward and sampler and the thinker's turn moves into the talker's
# decode frames; this leaves a gap between sampler and predictor with no added wait. c16 on the cell where 3p costs most, then c64,
# where 3p gained, to read whether the gain stays.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 A=/workspace/p3a
SEED="--seed 3451 --warmup 16"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1105_fp done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 card=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) c16 card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated_h100_config --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 "--bench-extra=$SEED" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
heavy() {
  local name=$1 tree=$2 card=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) c64 card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 64 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_a start"
boot xa_m1 $MAIN 0 & boot xa_p1 $A 1 & wait
boot xa_p2 $A 0 & boot xa_m2 $MAIN 1 & wait
boot xa_m3 $MAIN 0 & boot xa_p3 $A 1 & wait
boot xa_p4 $A 0 & boot xa_m4 $MAIN 1 & wait
heavy ha_m1 $MAIN 0 & heavy ha_a1 $A 1 & wait
heavy ha_a2 $A 0 & heavy ha_m2 $MAIN 1 & wait
log "q1105_a done"
