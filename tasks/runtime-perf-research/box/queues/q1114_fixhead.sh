#!/usr/bin/env bash
# 11-14, node-0000: the #3451 candidate without the decode mask fix (FIX 053a02a6f) on its own head: seeded c1 against id_m3,
# c16 and c64 against m16 and m64.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
FIX=/workspace/fix
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $FIX rev-parse --short HEAD) $(git -C $FIX rev-parse --short=12 HEAD:sglang_omni) $2 start"; }
log "q1114_fixhead start"
note id_h "seeded c1"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $FIX --model omni_colocated --card 0 --port 28400 --out $O/id_h \
  --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/id_h.driver.log 2>&1
log "boot id_h rc $?"
(cd $O/id_h/c1_clean/bench/audio && find . -name "*.wav" | sort | xargs sha256sum > $O/id_h/wav_sha256.txt)
log "identity $(python3 $K/wav_identity.py $O c1_clean id_m3 id_h 2>&1 | tail -1)"
note h16 "c16"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $FIX --model omni_colocated --card 0 --port 28400 --out $O/h16 \
  --loads 16 --passes warm,clean,events --bench-repeats 2 > $O/h16.driver.log 2>&1
log "boot h16 rc $?"
note h64 "c64"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $FIX --model omni_colocated --card 0 --port 28400 --out $O/h64 \
  --loads 64 --passes warm,events > $O/h64.driver.log 2>&1
log "boot h64 rc $?"
log "q1114_fixhead done"
