#!/usr/bin/env bash
# 11-04, node-0000 on two cards, the last of the lease: #3452's final form against main on the two-card speech default, the
# one layout it has not been read on. Single server at a time, boots m f f m, SeedTTS c64 and c16, warm and events. The first
# boot compiles for this config. Results are pulled as they land; the lease may end before the last boot.
set -u
O=/data/rp1104
K=/workspace/kit_t2/tasks/runtime-perf-research/box
MAIN=/workspace/main_debd337b4 F=/workspace/wk_final
export SGLANG_OMNI_STARTUP_TIMEOUT=2400
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
boot() {
  local name=$1 tree=$2
  log "boot $name $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni) two card start"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1104_twocard start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot td_m1 $MAIN
boot td_f1 $F
boot td_f2 $F
boot td_m2 $MAIN
log "q1104_twocard done"
