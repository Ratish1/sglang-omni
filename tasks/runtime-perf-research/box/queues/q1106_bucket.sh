#!/usr/bin/env bash
# 11-06, node-0000, after q1106_3451b: diagnostic only. On 0.5.22 the thinker's prefill had a 60 to 70 ms mode on 12 % of requests and the talker a 20 to 30 ms mode,
# both gone on 0.5.23, and #3451's +6.51 ms went with them. #3459 added the one-token bucket. These arms drop only that bucket. Does the tail return, does first
# audio, does #3451's cost? Each no-bucket arm is paired with its own form on the same cell.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
NM=/workspace/nbm NP=/workspace/nbp MAIN=/workspace/main
SEED="--seed 3451 --warmup 16"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_3451b done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 card=$3
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) card $card load $(tail -1 $O/cpu_load.log) start"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated_h100_config --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 "--bench-extra=$SEED" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1106_bucket start"
boot nm_a1 $MAIN 0 & boot nm_b1 $NM 1 & wait
boot nm_b2 $NM 0 & boot nm_a2 $MAIN 1 & wait
log "cell nm done"
boot np_a1 $NM 0 & boot np_b1 $NP 1 & wait
boot np_b2 $NP 0 & boot np_a2 $NM 1 & wait
boot np_a3 $NM 0 & boot np_b3 $NP 1 & wait
boot np_b4 $NP 0 & boot np_a4 $NM 1 & wait
log "q1106_bucket done"
