#!/usr/bin/env bash
# 11-14, node-0000, after q1114_tracec16: the final tree (ROWFINAL 0d20cda0a) at c64 with no flags, read against m64 and rs64.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
ROW=/workspace/rowready2
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_tracec16 done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_rowfinal64 start"
note rf64 $ROW "c64 default"
taskset -c $CARD_CPUS python3 $K/baseline.py --tree $ROW --model omni_colocated --card 0 --port 28400 --out $O/rf64 \
  --loads 64 --passes warm,events > $O/rf64.driver.log 2>&1
log "boot rf64 rc $?"
python3 $K/body_table.py $O c64 "row final c64 against main" main=m64 row=rf64 >> $O/queue.log 2>&1
log "q1114_rowfinal64 done"
