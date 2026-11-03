#!/usr/bin/env bash
# 11-03, node-0000: y cost the opening speech burst its stage loop (hop 3 to 75 ms over the first 64 requests at c64), six
# threads running Python at once. z holds two slots for a request's whole CPU time and gives one back only over a video's
# decode, so speech keeps two requests running as on the PR. SeedTTS boots w z z w, c64 and c16, warm and events, then
# Video-MME text-only and with audio, two boots each, against st_*, sy_*, sv_*, svy_* from the same node.
set -u
O=/data/rp1103
K=/workspace/kit_h/tasks/runtime-perf-research/box
W=/workspace/marks_wk Z=/workspace/wk_hold
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
SIX="--preprocessing.factory.max_concurrency 6"
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
wait_card() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 2000 ]; do
    log "waiting, card used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 30
  done
}
video() {
  local name=$1 bench=$2
  wait_card
  log "boot $name $(git -C $Z rev-parse --short HEAD) $(git -C $Z rev-parse --short=12 HEAD:sglang_omni) $bench start"
  python3 $K/baseline.py --tree $Z --model omni_colocated --card 0 --port 28000 --out $O/$name --bench $bench --loads 16 \
    --passes clean,events --serve-extra "$CONTEXT_32K $SIX" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech() {
  local name=$1 tree=$2; shift 2
  wait_card
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) seedtts start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28000 --out $O/$name --loads 64,16 \
    --passes warm,events "$@" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1103_wholdslot start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
speech sh_w1 $W
speech sh_z1 $Z --serve-extra "$SIX"
speech sh_z2 $Z --serve-extra "$SIX"
speech sh_w2 $W
video sz_z1 videomme_text
video sz_z2 videomme_text
video svz_z1 videomme
video svz_z2 videomme
log "q1103_wholdslot done"
