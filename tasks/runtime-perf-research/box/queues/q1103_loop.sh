#!/usr/bin/env bash
# 11-03, node-0000 on two cards: the single-loop form of the preprocessing stage. l is one event loop for every request's
# logic, N processor calls at once on private processors, media on the existing pool, no thread per request and no request
# bound (test/wk-loop-20261103); w is the PR's two whole-request workers on the marks branch. Same-time pairs, both card
# orders, one server a card. First a fill on each card, then Video-MME text-only and SeedTTS for w against l at N 2, then the
# processor count sweep on Video-MME text-only, N 1, 3, 4 against N 2. Read with preprocess_slots.py.
set -u
O=/data/rp1103n
K=/workspace/kit/tasks/runtime-perf-research/box
W=/workspace/marks_wk L=/workspace/wk_loop
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
fill() {
  local name=$1 card=$2 port=$3
  log "boot $name fill card $card start"
  python3 $K/baseline.py --tree $W --model omni_colocated --card $card --port $port --out $O/$name --loads 16 --passes warm > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 tree=$2 card=$3 port=$4 processors=$5 bench=${6:-videomme_text}
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $bench card $card N $processors start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --bench $bench --loads 16 \
    --passes clean,events --serve-extra "$CONTEXT_32K --preprocessing.factory.max_concurrency $processors" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) seedtts card $card start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1103_loop start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
fill fill_0 0 28400
fill fill_1 1 28800
video lt_w1 $W 0 28400 2 & video lt_l1 $L 1 28800 2 & wait
video lt_l2 $L 0 28400 2 & video lt_w2 $W 1 28800 2 & wait
speech ls_w1 $W 0 28400 & speech ls_l1 $L 1 28800 & wait
speech ls_l2 $L 0 28400 & speech ls_w2 $W 1 28800 & wait
log "q1103_loop pairs done"
video ln_1a $L 0 28400 1 & video ln_3a $L 1 28800 3 & wait
video ln_3b $L 0 28400 3 & video ln_1b $L 1 28800 1 & wait
video ln_4a $L 0 28400 4 & video ln_2a $L 1 28800 2 & wait
video ln_2b $L 0 28400 2 & video ln_4b $L 1 28800 4 & wait
log "q1103_loop done"
