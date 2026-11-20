#!/usr/bin/env bash
# 11-20, node-0000, after q1120_v3452: a second c64 block, SLOT (d3806fb6a) against main (de4cb9f0f), colocated speech
# same-time pairs on both cards, swapped, probed. The first block read first audio p99 +4.8 % and inter chunk p99
# +4.5 % with means level; a lean is read only when it repeats in a second block.
set -u
O=/data/v1120/slot2
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/tmain SLOT=/workspace/tslot
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
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
speech() {
  local name=$1 tree=$2 card=$3 load=$4 offset=$5
  note $name $tree "colocated c$load card $card probe"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads $load --passes warm,events --omni-chunk-audio --probe \
    --serve-env OMNI_HOP_PROBE_TALKER_BATCHES=1 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1120_v3452 done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1120_c64b start"
quiet_host
both "speech p5_s0 $SLOT 0 64 590" "speech p5_m1 $MAIN 1 64 590"
quiet_host
both "speech p6_m0 $MAIN 0 64 810" "speech p6_s1 $SLOT 1 64 810"
python3 $K/body_table.py $O c64 "slot against main c64, second block" main=p5_m1,p6_m0 slot=p5_s0,p6_s1 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "slot against main c64, both blocks" main=p3_m1,p4_m0,p5_m1,p6_m0 slot=p3_s0,p4_s1,p5_s0,p6_s1 \
  >> $O/queue.log 2>&1
python3 $K/first_audio_states.py --cell c64_events --main $O/p3_m1,$O/p4_m0,$O/p5_m1,$O/p6_m0 \
  --branch $O/p3_s0,$O/p4_s1,$O/p5_s0,$O/p6_s1 > $O/first_audio_states_c64.txt 2>&1
log "q1120_c64b done"
