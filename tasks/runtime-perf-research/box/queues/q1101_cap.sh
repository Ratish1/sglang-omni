#!/usr/bin/env bash
# 11-01, after q1101_wt: the talker cap (the talker's running cap follows the thinker's, 64; test/cap-on-main-20261031
# 14069e285 on e354ea41f) against main (talker cap 32), colocated, pinned, same-time pairs both card orders, SeedTTS EN
# streamed c64 and c16, warm, two clean passes. Read with pair_table, latency_paired, audio_fit and underrun (stalls).
set -u
O=/data/rp1101
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main_e354 CAP=/workspace/cap_14069e285
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
until grep -q "q1101_wt done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1101_cap start (cpus $(taskset -cp $$ | sed 's/.*: //'))"
boot() {
  local name=$1 tree=$2 card=$3 port=$4
  log "boot $name card $card port $port $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short HEAD:sglang_omni) start"
  python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name --bench-repeats 2 \
    --loads 64,16 --passes warm,clean > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
boot cp_m1 $MAIN 0 24000 & boot cp_c1 $CAP 1 24400 & wait
boot cp_c2 $CAP 0 24000 & boot cp_m2 $MAIN 1 24400 & wait
log "q1101_cap done"
