#!/usr/bin/env bash
# 11-05, node-0000, after q1105_td: the float32 reference for the Video-MME text lean. Colocated has no room for float32 weights
# (114 GB against a 101 GB budget), so this is the two-card default with the thinker alone on its card, bfloat16 against float32,
# the final form at c1 and c16. Decides whether load moves replies through rounding and what the model's own accuracy is.
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/wk_final
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
FP32="--thinker.engine.dtype float32"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1105_td done" $O/queue.log 2>/dev/null; do sleep 15; done
video() {
  local name=$1 extra=${2:-}
  log "boot $name two card loads 1,16 [$extra] start"
  python3 $K/baseline.py --tree $F --model omni_speech --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K $extra" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_fp start"
video fb1
video fd1 "$FP32"
video fd2 "$FP32"
video fb2
log "q1105_fp done"
