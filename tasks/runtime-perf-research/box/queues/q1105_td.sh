#!/usr/bin/env bash
# 11-05, node-0000, after q1105_3451: two-card speech, one server at a time on both cards. Main's preprocessing team of 112 threads
# oversubscribes libgomp's spin rule and burns 1 core; the final form's teams of 16 spin and burn 4.5. Passive reads whether that spin
# owns the two-card c64 first audio +1.3 to +1.6 % and req/s -1.2 to -3.9 %.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 F=/workspace/wk_final
PASSIVE="--preprocessing.env.OMP_WAIT_POLICY PASSIVE"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1105_3451 done" $O/queue.log 2>/dev/null; do sleep 15; done
boot() {
  local name=$1 tree=$2 extra=${3:-}
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) two card [$extra] start"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name --loads 64,16 --passes warm,events --serve-extra "$extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_td start"
boot tm1 $MAIN
boot tf1 $F
boot tp1 $F "$PASSIVE"
boot tp2 $F "$PASSIVE"
boot tf2 $F
boot tm2 $MAIN
log "q1105_td done"
