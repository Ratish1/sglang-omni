#!/usr/bin/env bash
# 11-14, node-0000, after q1114_eqload: the #3451 candidate FIX with the existing early start (partial start and the
# one-chunk start topology), c16 and c64 speech, events pass. Proof for per-row talker readiness: first audio by state
# against h16/h64 (FIX, build at stream done) and how many talker decode steps the whole-batch gate defers.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
FIX=/workspace/fix
EARLY="--talker_ar.factory.enable_partial_start true --talker_ar.factory.enable_talker_start_topology true"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_eqload done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_earlystart start"
speech() {
  local name=$1 load=$2 passes=$3
  note $name $FIX "c$load early start"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $FIX --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads $load --serve-extra "$EARLY" --passes $passes > $O/$name.driver.log 2>&1
  log "boot $name rc $? gate $(grep -c 'talker chunk gate' $O/$name/serve.log) $(grep 'talker chunk gate' $O/$name/serve.log | tail -1 | cut -c1-160)"
}
speech es16 16 warm,events
speech es64 64 warm,events
for base in m h; do
  python3 $K/first_audio_states.py --main $O/${base}16 --branch $O/es16 >> $O/queue.log 2>&1
  python3 $K/first_audio_states.py --cell c64_events --main $O/${base}64 --branch $O/es64 >> $O/queue.log 2>&1
done
python3 $K/body_table.py $O c16 "early start c16" main=m16 fix=h16 early=es16 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "early start c64" main=m64 fix=h64 early=es64 >> $O/queue.log 2>&1
log "q1114_earlystart done"
