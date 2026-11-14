#!/usr/bin/env bash
# 11-14, node-0000, after q1114_rowready: the per-row readiness head with the colocated talker starting at one chunk by
# default (ROWFINAL 0d20cda0a, adds the aborted-row finish and the config default to the measured tree). Units; seeded c1
# against main's hashes with no flags; c16 with no flags, read against rs16 (same path through flags) and m16.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
ROW=/workspace/rowready2
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_rowready done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_rowfinal start"
log "units ROWFINAL start"
(cd $ROW && taskset -c $CARD_CPUS python3 -m pytest tests/unit_test/qwen3_omni -q -p no:cacheprovider > $O/units_rowfinal.log 2>&1)
log "units ROWFINAL rc $? $(grep -E 'passed|failed' $O/units_rowfinal.log | tail -1)"
note id_rf $ROW "seeded c1 default"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $ROW --model omni_colocated --card 0 --port 28400 --out $O/id_rf \
  --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/id_rf.driver.log 2>&1
log "boot id_rf rc $?"
(cd $O/id_rf/c1_clean/bench/audio && find . -name "*.wav" | sort | xargs sha256sum > $O/id_rf/wav_sha256.txt)
log "identity $(python3 $K/wav_identity.py $O c1_clean id_m3 id_rf 2>&1 | tail -1)"
note rf16 $ROW "c16 default"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $ROW --model omni_colocated --card 0 --port 28400 --out $O/rf16 \
  --loads 16 --passes warm,events > $O/rf16.driver.log 2>&1
log "boot rf16 rc $? waiting $(grep -c 'talker rows waiting' $O/rf16/serve.log)"
python3 $K/body_table.py $O c16 "row final c16 against main" main=m16 row=rf16 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c16 "row final c16 against row flags" flags=rs16 row=rf16 >> $O/queue.log 2>&1
log "q1114_rowfinal done"
