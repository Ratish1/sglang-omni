#!/usr/bin/env bash
# 11-23, node-0001, both cards. Fix A's c64 lean read on its landing base: fix A 3491dc8a7 against upstream main
# 22bf41ef5, speech alone c64 colocated, same-time pairs on the two cards, each server on its own socket, swapped,
# after a same-tree pair (main on both cards) for the band. The 11-22 card 0 pairs read latency median +1.3 / +0.3 %
# and RTF +0.5 / +0.2 % against main in both pairs, every other c64 row level or better.
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
UMAIN=/workspace/umain FIXA=/workspace/fixa
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((28000 + 2000 * $1 + $2)); }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), node0 $(numastat -m 2>/dev/null | awk '/MemFree/ {print $2}') MB free"
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
  note $name $tree "colocated c$load card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads $load --passes warm,events --omni-chunk-audio \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "setup done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_a start"
quiet_host
both "speech aa_m0 $UMAIN 0 64 110" "speech aa_m1 $UMAIN 1 64 110"
quiet_host
both "speech a1_m0 $UMAIN 0 64 130" "speech a1_f1 $FIXA 1 64 130"
quiet_host
both "speech a2_f0 $FIXA 0 64 150" "speech a2_m1 $UMAIN 1 64 150"
python3 $K/body_table.py $O c64 "main against main, same time, card 0 / card 1" m0=aa_m0 m1=aa_m1 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "fix A against main, same-time pairs, swapped" main=a1_m0,a2_m1 fix=a1_f1,a2_f0 \
  >> $O/queue.log 2>&1
python3 $K/first_audio_states.py --cell c64_events --main $O/a1_m0,$O/a2_m1 --branch $O/a1_f1,$O/a2_f0 \
  > $O/a_states_c64.txt 2>&1
grep -E "server first audio|admission|talker|thinker|frames until" $O/a_states_c64.txt >> $O/queue.log
log "A read"
log "q1123_a done"
