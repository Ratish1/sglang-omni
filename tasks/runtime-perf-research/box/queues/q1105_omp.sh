#!/usr/bin/env bash
# 11-05, node-0000, two H200 on separate sockets, each server pinned to its card's socket. Fills both configs, then #3452's
# c16 lean: main, the final form, the final form with passive OpenMP in preprocessing, same-time, every arm on both cards.
# The lean reads as thinker reply and talker pickup and is absent on two cards; passive reads whether spin owns it.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 F=/workspace/wk_final
PASSIVE="--preprocessing.env.OMP_WAIT_POLICY PASSIVE"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CPUS0=0-55,112-167 CPUS1=56-111,168-223
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
fill() {
  local name=$1 model=$2 card=$3
  log "fill $name $model card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $MAIN --model $model --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm > $O/$name.driver.log 2>&1
  log "fill $name rc $?"
}
boot() {
  local name=$1 arm=$2 card=$3 tree=$MAIN extra=""
  case $arm in
    m) ;;
    f) tree=$F ;;
    p) tree=$F extra=$PASSIVE ;;
  esac
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) arm $arm card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_omp start"
fill fill_a omni_colocated 0 & fill fill_b omni_colocated_h100_config 1 & wait
boot om1 m 0 & boot of1 f 1 & wait
boot op1 p 0 & boot om2 m 1 & wait
boot of2 f 0 & boot op2 p 1 & wait
boot of3 f 0 & boot om3 m 1 & wait
boot om4 m 0 & boot op3 p 1 & wait
boot op4 p 0 & boot of4 f 1 & wait
log "q1105_omp done"
