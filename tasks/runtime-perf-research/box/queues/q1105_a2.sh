#!/usr/bin/env bash
# 11-05, node-0000, after q1105_fp2: the rest of #3451's arm b6309c620 against main. Seeded c1 byte identity first, since the arm
# only moves a wait. Then the default h200 config, the unseeded h100_bf16 config, and two-card c64 and c16.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 A=/workspace/p3a
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1105_fp2 done" $O/queue.log 2>/dev/null; do sleep 15; done
identity() {
  local name=$1 tree=$2 card=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) seeded c1 card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
boot() {
  local name=$1 tree=$2 card=$3 model=$4
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $model card $card start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model $model --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
cell() {
  local tag=$1 model=$2
  boot ${tag}_m1 $MAIN 0 $model & boot ${tag}_p1 $A 1 $model & wait
  boot ${tag}_p2 $A 0 $model & boot ${tag}_m2 $MAIN 1 $model & wait
  boot ${tag}_m3 $MAIN 0 $model & boot ${tag}_p3 $A 1 $model & wait
  boot ${tag}_p4 $A 0 $model & boot ${tag}_m4 $MAIN 1 $model & wait
  log "cell $tag done"
}
twocard() {
  local name=$1 tree=$2
  log "boot $name $(git -C $tree rev-parse --short HEAD) two card start"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_a2 start"
identity ia_m1 $MAIN 0 & identity ia_a1 $A 1 & wait
cell xah omni_colocated
cell xab omni_colocated_h100_config
twocard ta_m1 $MAIN
twocard ta_a1 $A
twocard ta_a2 $A
twocard ta_m2 $MAIN
log "q1105_a2 done"
