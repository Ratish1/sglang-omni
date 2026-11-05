#!/usr/bin/env bash
# 11-06, node-0000, after q1106_bisect. The candidate on two cards on the new base, boots m a a m, full host, the row #3451's PR form already has as tp_*.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main A=/workspace/p3a
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1106_bisect done" $O/queue.log 2>/dev/null; do sleep 15; done
twocard() {
  local name=$1 tree=$2
  log "boot $name $(git -C $tree rev-parse --short HEAD) two card load $(tail -1 $O/cpu_load.log) start"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1106_twocard start"
twocard ta_m1 $MAIN
twocard ta_a1 $A
twocard ta_a2 $A
twocard ta_m2 $MAIN
log "q1106_twocard done"
