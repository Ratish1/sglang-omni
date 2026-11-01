#!/usr/bin/env bash
# 11-01, after q1101_bis2: the cap with 3p, 3s and mixed admissions, without the request build
# (test/cap-3p-mix-20261101 f3ea87b22; the cap stack 3c6ceb520 minus the request build), against main, same-time pairs
# both card orders, colocated, pinned, c64 and c16 warm + two clean passes + an events pass. Read against the cap stack
# (cs_*): stalls, rows, and the c16 first audio states.
set -u
O=/data/rp1101
K=/workspace/kit_mix3/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 MIX=/workspace/c3pmix_f3ea87b22
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_bis2 done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1101_mix3 start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --bench-repeats 2 \
    --loads 64,16 --passes warm,clean,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
boot mm_m1 $MAIN 0 24000 & boot mm_x1 $MIX 1 24400 & wait
boot mm_x2 $MIX 0 24000 & boot mm_m2 $MAIN 1 24400 & wait
log "q1101_mix3 done"
