#!/usr/bin/env bash
# 11-15, node-0000, both H200s: #3452 (LANE ecfebb1f3) against main 35890f10c on the CI's Video-MME server, the two-card
# disaggregated layout (thinker and encoders card 0, talker and code2wav card 1, examples/run_qwen3_omni_speech_server.py
# with the CI's flags), text Video-MME CI sample. The layout where the OpenMP split applies: pinned to card 0's socket,
# 112 CPUs, main's preprocessing team 112 against 14 per video thread, 16 FFmpeg threads a video against 14. c1 (one
# video at a time) and c16, two clean repeats and an events pass each, boots m l l m; each pass records each process's
# resident peak (memory.json).
set -u
O=/data/v1115
K=/workspace/kit2/tasks/runtime-perf-research/box
LANE=/workspace/vlane MAIN=/workspace/main
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
twocard() {
  local name=$1 tree=$2
  quiet_host
  note $name $tree "two-card ci video c1 c16"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_disagg_ci --port 28400 --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1115_twocard start"
twocard tc_m1 $MAIN
twocard tc_l1 $LANE
twocard tc_l2 $LANE
twocard tc_m2 $MAIN
log "q1115_twocard done"
