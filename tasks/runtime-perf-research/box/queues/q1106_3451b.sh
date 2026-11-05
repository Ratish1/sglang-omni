#!/usr/bin/env bash
# 11-06, node-0000, after q1106_lane: on SGLang 0.5.23 #3451 as it stands read -1.77 ms on the cell where it cost +6.51 ms on 0.5.22, level with the
# candidate and the bisect arm. So: the PR as it stands against main on the default h200 config and the rest of the old cross, directly against
# the candidate, then c64 for both, then two-card. Complete events on every boot, for the states.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main P=/workspace/p3 A=/workspace/p3a
SEED="--seed 3451 --warmup 16"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_lane done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 card=$3 model=$4 bench=${5:-}
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $model [$bench] card $card load $(tail -1 $O/cpu_load.log) start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model $model --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 "--bench-extra=$bench" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
pairs() {
  local tag=$1 first=$2 second=$3 count=$4 model=$5 bench=${6:-}
  local index=1
  while [ $index -le $count ]; do
    if [ $((index % 2)) = 1 ]; then
      boot ${tag}_m$index $first 0 $model "$bench" & boot ${tag}_p$index $second 1 $model "$bench" & wait
    else
      boot ${tag}_p$index $second 0 $model "$bench" & boot ${tag}_m$index $first 1 $model "$bench" & wait
    fi
    index=$((index + 1))
  done
  log "cell $tag done"
}
heavy() {
  local name=$1 tree=$2 card=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) c64 card $card load $(tail -1 $O/cpu_load.log) start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 64 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
twocard() {
  local name=$1 tree=$2
  log "boot $name $(git -C $tree rev-parse --short HEAD) two card load $(tail -1 $O/cpu_load.log) start"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1106_3451b start"
pairs ph $MAIN $P 4 omni_colocated
pairs pb $MAIN $P 4 omni_colocated_h100_config "$SEED"
pairs pa $P $A 4 omni_colocated
heavy cp_m1 $MAIN 0 & heavy cp_p1 $P 1 & wait
heavy cp_p2 $P 0 & heavy cp_m2 $MAIN 1 & wait
heavy ca_m1 $MAIN 0 & heavy ca_a1 $A 1 & wait
heavy ca_a2 $A 0 & heavy ca_m2 $MAIN 1 & wait
log "cell c64 done"
twocard tp_m1 $MAIN
twocard tp_p1 $P
twocard tp_p2 $P
twocard tp_m2 $MAIN
log "q1106_3451b done"
