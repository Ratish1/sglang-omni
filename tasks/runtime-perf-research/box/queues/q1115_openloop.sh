#!/usr/bin/env bash
# 11-15, node-0000, after q1115_syncfree: SeedTTS alone on one seeded open-loop arrival schedule (OMNI_ARRIVAL_SEED, 12
# requests a second, 64 allowed in flight so no cap binds), colocated same-time pairs on both cards, swapped. The closed
# c16 loop holds 16 in flight, so a talker that keeps each request longer holds more rows at once; equal arrivals below
# capacity read each stream's chunk cadence at the same offered work.
set -u
O=/data/rp1115
K=/workspace/kit2/tasks/runtime-perf-research/box
FIX=/workspace/fix FOLD=/workspace/fold FOLDSYNC=/workspace/foldsync
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
CPUS0=0-55,112-167
CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1115_syncfree done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1115_openloop start"
openloop() {
  local name=$1 tree=$2 card=$3
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  note $name $tree "colocated 12/s seed $OMNI_ARRIVAL_SEED card $card probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --passes warm,events --omni-chunk-audio --probe --serve-env "OMNI_HOP_PROBE_TALKER_BATCHES=1" \
    "--bench-extra=--request-rate=12" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
both "openloop o1_x0 $FIX 0" "openloop o1_s1 $FOLDSYNC 1"
both "openloop o2_s0 $FOLDSYNC 0" "openloop o2_x1 $FIX 1"
both "openloop o3_f0 $FOLD 0" "openloop o3_x1 $FIX 1"
python3 $K/chunk_gap_parts.py --cell c64_events $O/o1_x0 $O/o2_x1 $O/o3_x1 $O/o1_s1 $O/o2_s0 $O/o3_f0 > $O/chunk_gap_parts_O.txt 2>&1
log "q1115_openloop done"
