#!/usr/bin/env bash
# 11-23, node-0001, both cards, after q1123_a2. Fix A's c16 lean read on its landing base. The card 1 B A A B (beside
# card 0's video pairs) read QPS -0.5 / -0.4 % and latency median +0.9 / +0.1 % in both pairs. Same-time pairs on the
# two cards, swapped, after main on both cards for the band, then the per-state split and the chunk gaps.
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
UMAIN=/workspace/umain FIXA=/workspace/fixa
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
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
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "colocated c16 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1123_a2 done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_a3 start"
quiet_host
both "speech ab_m0 $UMAIN 0 170" "speech ab_m1 $UMAIN 1 170"
quiet_host
both "speech a3_m0 $UMAIN 0 190" "speech a3_f1 $FIXA 1 190"
quiet_host
both "speech a4_f0 $FIXA 0 610" "speech a4_m1 $UMAIN 1 610"
python3 $K/body_table.py $O c16 "main against main c16, same time, card 0 / card 1" m0=ab_m0 m1=ab_m1 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c16 "fix A against main c16, same-time pairs, swapped" main=a3_m0,a4_m1 fix=a3_f1,a4_f0 \
  >> $O/queue.log 2>&1
python3 $K/first_audio_states.py --cell c16_events --main $O/a3_m0,$O/a4_m1 --branch $O/a3_f1,$O/a4_f0 \
  > $O/a3_states_c16.txt 2>&1
grep -E "server first audio|hop to the thinker|thinker build|thinker prefill|thinker chunks|talker build|frames until" \
  $O/a3_states_c16.txt >> $O/queue.log
python3 $K/first_chunk_gap.py --cell c16_events $O main=a3_m0,a4_m1 fix=a3_f1,a4_f0 2>&1 | grep gaps >> $O/queue.log
log "A3 read"
log "q1123_a3 done"
