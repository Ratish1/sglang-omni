#!/usr/bin/env bash
# 11-05, node-0000, after q1105_cand: the float32 reference again. Its weights take 114 GB of the thinker's card, so the static
# pool needs a share above the automatic 0.856 (the boot's own message gives the minimum).
set -u
O=/data/rp1105
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/wk_final
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
FP32="--thinker.engine.dtype float32 --thinker.engine.mem_fraction_static 0.95"
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1105_cand done" $O/queue.log 2>/dev/null; do sleep 15; done
video() {
  local name=$1
  log "boot $name two card loads 1,16 [$FP32] start"
  python3 $K/baseline.py --tree $F --model omni_speech --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K $FP32" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1105_fp2 start"
video fe1
video fe2
log "q1105_fp2 done"
