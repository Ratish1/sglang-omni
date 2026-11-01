#!/usr/bin/env bash
# 11-01, after q1101_mix3: the cap stack's c16 first audio (+2.17 % paired, cs_*) by request state. The cap stack
# (test/cap-stack-20261101 3c6ceb520) against main, same-time pairs both card orders, colocated, pinned, c16 warm then an
# events pass; read with first_audio_states.py.
set -u
O=/data/rp1101
K=/workspace/kit_csev/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 STACK=/workspace/cstack_3c6ceb520
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_mix3 done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1101_csev start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
boot ev_m1 $MAIN 0 24000 & boot ev_s1 $STACK 1 24400 & wait
boot ev_s2 $STACK 0 24000 & boot ev_m2 $MAIN 1 24400 & wait
taskset -c 0-55 python3 $K/first_audio_states.py --cell c16_events --main ev_m1,ev_m2 --branch ev_s1,ev_s2 > $O/csev_read.txt 2>&1
log "q1101_csev done"
