#!/usr/bin/env bash
# 11-04, after q1104_twocard, node-0000: same-time A/A at c16, the band #3452's c16 events first audio (+8.3, +2.6 ms) is read
# against. Main beside main, then the final form beside itself, colocated, SeedTTS c16, warm and events.
set -u
O=/data/rp1104
K=/workspace/kit_aa/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 F=/workspace/wk_final
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1104_twocard done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) a/a card $card start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --loads 16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1104_aa start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot aa_m1 $MAIN 0 28400 & boot aa_m2 $MAIN 1 28800 & wait
boot aa_f1 $F 0 28400 & boot aa_f2 $F 1 28800 & wait
log "q1104_aa done"
