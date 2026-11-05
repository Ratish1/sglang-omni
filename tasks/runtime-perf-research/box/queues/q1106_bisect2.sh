#!/usr/bin/env bash
# 11-06, node-0000, after q1106_nopassive. The bisect arm, the Qwen3-TTS order with no added wait, read -1.18 ms (se 0.65) on h200 unseeded, where the PR form read +2.34. It only
# reorders a wait, so seeded c1 output must be byte identical. Then the rows the other forms have: h200 c64 and two-card.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main S=/workspace/p3s
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_nopassive done" $O/queue.log 2>/dev/null; do sleep 15; done
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 load $(tail -1 $O/cpu_load.log) start"; }
identity() {
  local name=$1 tree=$2 card=$3
  note $name $tree "seeded c1 card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
heavy() {
  local name=$1 tree=$2 card=$3
  note $name $tree "c64 card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 64 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
twocard() {
  local name=$1 tree=$2
  note $name $tree "two card"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1106_bisect2 start"
identity is_m1 $MAIN 0 & identity is_s1 $S 1 & wait
log "cell is done"
heavy cs_m1 $MAIN 0 & heavy cs_s1 $S 1 & wait
heavy cs_s2 $S 0 & heavy cs_m2 $MAIN 1 & wait
log "cell cs done"
twocard ts_m1 $MAIN
twocard ts_s1 $S
twocard ts_s2 $S
twocard ts_m2 $MAIN
log "q1106_bisect2 done"
