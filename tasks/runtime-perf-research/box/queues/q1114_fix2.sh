#!/usr/bin/env bash
# 11-14, node-0000, after q1114_fix: #3451 with the sync-free prompt build and the 11-13 decode mask fix (FIX2 ee2952973), the
# decode fix that alone cost first audio +3.1 % at c16. Seeded c1 against id_m3, c16 and c64 against m16, f16, m64, f64.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
FIX2=/workspace/fix2
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_fix done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_fix2 start"
log "units FIX2 start"
(cd $FIX2 && taskset -c $CARD_CPUS python3 -m pytest tests/unit_test/qwen3_omni -q -p no:cacheprovider > $O/units_fix2.log 2>&1)
log "units FIX2 rc $? $(tail -1 $O/units_fix2.log)"
note id_f2 $FIX2 "seeded c1"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $FIX2 --model omni_colocated --card 0 --port 28400 --out $O/id_f2 \
  --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/id_f2.driver.log 2>&1
log "boot id_f2 rc $?"
(cd $O/id_f2/c1_clean/bench/audio && find . -name "*.wav" | sort | xargs sha256sum > $O/id_f2/wav_sha256.txt)
log "identity $(python3 $K/wav_identity.py $O c1_clean id_m3 id_f2 2>&1 | tail -1)"
speech() {
  local name=$1 load=$2 passes=$3 repeats=$4
  note $name $FIX2 "c$load"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $FIX2 --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads $load --passes $passes --bench-repeats $repeats > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech g16 16 warm,clean,events 2
speech g64 64 warm,events 1
log "q1114_fix2 done"
