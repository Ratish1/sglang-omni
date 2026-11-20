#!/usr/bin/env bash
# 11-20, node-0000, after q1120_slot: SeedTTS alone on one seeded open-loop arrival schedule (OMNI_ARRIVAL_SEED, 12
# requests a second, 64 allowed in flight so no cap binds), SLOT (d3806fb6a) against main (de4cb9f0f), colocated
# same-time pairs on both cards, swapped. The closed c16 loop holds 16 in flight, so a talker that frees its rows sooner
# keeps more of the 16 in their thinker phase at once; equal arrivals below capacity read each request's first audio and
# chunk cadence at the same offered work.
set -u
O=/data/v1120/slot2
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/tmain SLOT=/workspace/tslot
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((28000 + 2000 * $1 + $2)); }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
openloop() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "colocated 12/s seed $OMNI_ARRIVAL_SEED card $card probe"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 64 --passes warm,events --omni-chunk-audio --probe \
    --serve-env OMNI_HOP_PROBE_TALKER_BATCHES=1 "--bench-extra=--request-rate=12" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1120_slot done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1120_slot_openloop start"
quiet_host
both "openloop o1_s0 $SLOT 0 1950" "openloop o1_m1 $MAIN 1 1950"
quiet_host
both "openloop o2_m0 $MAIN 0 2050" "openloop o2_s1 $SLOT 1 2050"
python3 $K/body_table.py $O c64 "slot against main, 12/s open loop" main=o1_m1,o2_m0 slot=o1_s0,o2_s1 >> $O/queue.log 2>&1
python3 $K/first_audio_states.py --cell c64_events --main $O/o1_m1,$O/o2_m0 --branch $O/o1_s0,$O/o2_s1 \
  > $O/first_audio_states_open.txt 2>&1
python3 $K/chunk_gap_parts.py --cell c64_events $O/o1_s0 $O/o2_s1 $O/o1_m1 $O/o2_m0 > $O/chunk_gap_parts_open.txt 2>&1
log "q1120_slot_openloop done"
