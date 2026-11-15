#!/usr/bin/env bash
# 11-15, node-0000, after q1115_rootcause: card 0 runs traced colocated c16 (nsys cuda, NVTX, OS runtime, gpuctxsw), main
# then FOLD, one context switch session at a time, read by talker_steps.py (per talker step: time in syncs, host time,
# the talker's own device time and other processes' device time inside it). Card 1 meanwhile reruns the cells of the void
# pair p3 (FIX, then FOLD) with the probe.
set -u
O=/data/rp1115
K=/workspace/kit2/tasks/runtime-perf-research/box
NV=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts/omni_pipeline_nvtx
MAIN=/workspace/main FIX=/workspace/fix FOLD=/workspace/fold
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1115_rootcause done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1115_followup start"
traced() {
  local name=$1 tree=$2
  note $name $tree "colocated c16 card 0 nsys"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,nsys --nsys-session $name --nsys-trace cuda,nvtx,osrt --nsys-graph-trace graph --nsys-gpuctxsw \
    --nsys-delay 20 --nsys-seconds 15 --serve-env "OMNI_PIPE_NVTX=1 PYTHONPATH=$NV:$tree" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
  (cd $O/$name && python3 - c16_nsys/trace.sqlite serve.log < $K/talker_steps.py > talker_steps.txt 2>&1)
  log "talker steps $name rc $?"
}
colocated() {
  local name=$1 tree=$2
  note $name $tree "colocated c16 card 1 probe"
  taskset -c $CPUS1 python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port 30400 --out $O/$name \
    --loads 16 --passes warm,events --omni-chunk-audio --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
(traced tc_m $MAIN; traced tc_f $FOLD) &
card0=$!
(colocated p6_x1 $FIX; colocated p6_f1 $FOLD) &
card1=$!
wait $card0 $card1
log "q1115_followup done"
