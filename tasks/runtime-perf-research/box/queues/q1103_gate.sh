#!/usr/bin/env bash
# 11-03, after q1103_loop, node-0000 on two cards: g is the single loop with a gate in arrival order, entered at a request's
# start or, for a video, after its decode (test/wk-loop-20261103). The single loop alone finished a burst's work sooner than
# two workers (last preprocess_end 322 against 342 ms) but lost arrival order and routing (first 64 first audio +66 ms). Same-time
# pairs against w, both card orders, on SeedTTS, Video-MME text-only and with audio. Peak memory is in preprocess_end's metadata.
set -u
O=/data/rp1103n
K=/workspace/kit_g/tasks/runtime-perf-research/box
W=/workspace/marks_wk G=/workspace/wk_gate
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
speech() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) seedtts card $card start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 tree=$2 card=$3 port=$4 bench=$5
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) $bench card $card start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --bench $bench --loads 16 \
    --passes clean,events --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "q1103_loop done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1103_gate start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
speech gs_w1 $W 0 28400 & speech gs_g1 $G 1 28800 & wait
speech gs_g2 $G 0 28400 & speech gs_w2 $W 1 28800 & wait
video gt_w1 $W 0 28400 videomme_text & video gt_g1 $G 1 28800 videomme_text & wait
video gt_g2 $G 0 28400 videomme_text & video gt_w2 $W 1 28800 videomme_text & wait
video gv_w1 $W 0 28400 videomme & video gv_g1 $G 1 28800 videomme & wait
video gv_g2 $G 0 28400 videomme & video gv_w2 $W 1 28800 videomme & wait
log "q1103_gate done"
