#!/usr/bin/env bash
# 11-01, after q1101_wk: preprocessing workers bounded (94abfaaa2) against main, colocated, pinned, Video-MME CI
# text-only at c16 and the 32k context, two passes per boot, same-time pairs both card orders.
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 WK=/workspace/wk_94abfaaa2
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_wk done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1101_wt start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
boot() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --bench-repeats 2 \
    --bench videomme_text --loads 16 --passes clean --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
boot wt_m1 $MAIN 0 24000 & boot wt_w1 $WK 1 24400 & wait
boot wt_w2 $WK 0 24000 & boot wt_m2 $MAIN 1 24400 & wait
log "q1101_wt done"
