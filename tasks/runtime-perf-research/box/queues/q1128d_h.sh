#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_g: where the running video limit stops paying, on DEV2 (4cac082f9). CI one
# card, CI Video-MME text c16 x2 from a fresh boot (pass 1 prefills every video, pass 2 reads prefix hits), with
# --preprocessing.factory.max_running_video_requests 2, 4 and 8, same-time pairs against 8, cards swapped. Little's law
# puts the need near the thinker's video rate times a video's preprocessing time (about 3/s x 1 s); host memory grows
# about 1.3 GB per running video.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
V=/workspace/dev2
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
until grep -q "q1128d_g done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_h start"
limit() {
  local name=$1 card=$2 offset=$3 videos=$4
  note $name $V "CI one card, max_running_video_requests $videos, text c16 x2 from a fresh boot, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $V --model omni_ci_onecard --card $card \
    --port $(port_of $card $offset) --out $O/$name \
    --serve-extra "--preprocessing.factory.max_running_video_requests $videos" --bench videomme_text --loads 16 \
    --passes clean --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(python3 -c "import json,glob; print([json.load(open(p))['speed']['throughput_qps'] for p in sorted(glob.glob('$O/$name/c16_clean/bench/rep*/videomme_results.json'))])")"
}
both "limit lv2a 0 6100 2" "limit lv8a 1 6100 8"
both "limit lv8b 0 6300 8" "limit lv2b 1 6300 2"
both "limit lv4a 0 6500 4" "limit lv8c 1 6500 8"
both "limit lv8d 0 6700 8" "limit lv4b 1 6700 4"
log "limit read"
log "q1128d_h done"
