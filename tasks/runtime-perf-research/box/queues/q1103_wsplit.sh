#!/usr/bin/env bash
# 11-03, after q1103_wslots, node-0000: media outside the processor bound. y is six request threads with two processor
# slots (test/wk-slots-20261103), x four whole-request workers, w the PR's two (both on the marks branch). Video-MME text-only
# and with audio, c16, 32k context, a clean pass then an events pass, boots y x x y; with audio against sv_w and sv_x from
# q1103_wslots. Then SeedTTS, boots w y y w, c64 and c16, warm and events: speech has no media wait, so its rows are the check
# that the form costs speech nothing. Read with preprocess_slots.py.
set -u
O=/data/rp1103
K=/workspace/kit_s/tasks/runtime-perf-research/box
W=/workspace/marks_wk Y=/workspace/wk_slots
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
FOUR="--preprocessing.factory.max_concurrency 4"
SIX="--preprocessing.factory.max_concurrency 6"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
video() {
  local name=$1 tree=$2 bench=$3 extra=$4
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $bench [$extra] start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 27600 --out $O/$name --bench $bench --loads 16 \
    --passes clean,events --serve-extra "$CONTEXT_32K $extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech() {
  local name=$1 tree=$2 extra=$3
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) seedtts [$extra] start"
  if [ -n "$extra" ]; then
    python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 27600 --out $O/$name --loads 64,16 \
      --passes warm,events --serve-extra "$extra" > $O/$name.driver.log 2>&1
  else
    python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 27600 --out $O/$name --loads 64,16 \
      --passes warm,events > $O/$name.driver.log 2>&1
  fi
  log "boot $name rc $?"
}
until grep -q "q1103_wslots done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1103_wsplit start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
video sy_y1 $Y videomme_text "$SIX"
video sy_x1 $W videomme_text "$FOUR"
video sy_x2 $W videomme_text "$FOUR"
video sy_y2 $Y videomme_text "$SIX"
video svy_y1 $Y videomme "$SIX"
video svy_y2 $Y videomme "$SIX"
speech ss_w1 $W ""
speech ss_y1 $Y "$SIX"
speech ss_y2 $Y "$SIX"
speech ss_w2 $W ""
cd $O
taskset -c 0-55 python3 $K/preprocess_slots.py sy_y1/c16_events,sy_y2/c16_events sy_x1/c16_events,sy_x2/c16_events \
  svy_y1/c16_events,svy_y2/c16_events ss_w1/c16_events,ss_w2/c16_events ss_y1/c16_events,ss_y2/c16_events > $O/read_split.txt 2>&1
log "q1103_wsplit done"
