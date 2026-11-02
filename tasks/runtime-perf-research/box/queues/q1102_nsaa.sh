#!/usr/bin/env bash
# 11-02, node-0001, after q1102_3p: main debd337b4 against itself, colocated, pinned, same time on both cards, c16
# non-streamed warm + two clean passes, two rounds of fresh boots. The band for 3p's non-streamed row (ns_*: QPS -1.9 %,
# carried by one main boot, ns_m2 at 15.2 req/s against 14.7 to 14.8 for the other three): is 15.2 a card property, a boot
# state, or where main sits. Each round starts only with both cards free; other sessions share this node.
set -u
O=/data/rp1102
K=/workspace/kit_aa/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cards_free() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 2000 ]
}
wait_cards() {
  until cards_free; do
    log "nsaa waiting, cards used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 60
  done
}
until grep -q "q1102_3p done" $O/queue.log 2>/dev/null; do sleep 15; done
log "q1102_nsaa start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot() {
  local name=$1 card=$2 port=$3
  log "boot $name card $card port $port $(git -C $MAIN rev-parse --short HEAD) $(git -C $MAIN rev-parse --short=12 HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $MAIN --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,clean --bench-repeats 2 --no-stream > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
wait_cards
boot na_c0 0 25000 & boot na_c1 1 25400 & wait
wait_cards
boot nb_c0 0 25000 & boot nb_c1 1 25400 & wait
log "q1102_nsaa done"
