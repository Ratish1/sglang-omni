#!/usr/bin/env bash
# 11-03, after q1103_marks, node-0000: #3452's form on video. Its head under preprocessing marks, Video-MME CI text-only and
# with audio, c16, 32k context, a clean pass then an events pass per boot, boots w x x w. w is the PR's two workers, x
# is four: whole requests at four hold at least as much processor and media concurrency as media outside two processor
# slots, so x bounds what moving media out can gain. Read with preprocess_slots.py.
set -u
O=/data/rp1103
K=/workspace/kit_m/tasks/runtime-perf-research/box
W=/workspace/marks_wk
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
FOUR="--preprocessing.factory.max_concurrency 4"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
boot() {
  local name=$1 bench=$2 extra=$3
  wait_card
  log "boot $name $(git -C $W rev-parse --short HEAD) $(git -C $W rev-parse --short=12 HEAD:sglang_omni) $bench [$extra] start"
  python3 $K/baseline.py --tree $W --model omni_colocated --card 0 --port 27200 --out $O/$name --bench $bench --loads 16 \
    --passes clean,events --serve-extra "$CONTEXT_32K $extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1103_marks done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1103_wslots start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot st_w1 videomme_text ""
boot st_x1 videomme_text "$FOUR"
boot st_x2 videomme_text "$FOUR"
boot st_w2 videomme_text ""
boot sv_w1 videomme ""
boot sv_x1 videomme "$FOUR"
boot sv_x2 videomme "$FOUR"
boot sv_w2 videomme ""
cd $O
taskset -c 0-55 python3 $K/preprocess_slots.py st_w1/c16_events,st_w2/c16_events st_x1/c16_events,st_x2/c16_events \
  sv_w1/c16_events,sv_w2/c16_events sv_x1/c16_events,sv_x2/c16_events > $O/read_slots.txt 2>&1
log "q1103_wslots done"
