#!/usr/bin/env bash
# 11-15, node-0000, after q1115_share2: main 35890f10c in the CI two-card speech server's SeedTTS c16 beside 4 looping
# videos (whole host), two boots, the base for md*_l (previous head) and md*_b (the video CPU share).
set -u
O=/data/v1115
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
ALL=0-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
until grep -q "q1115_share2 done" $O/queue.log; do sleep 10; done
mixed() {
  local name=$1 tree=$2
  note $name $tree "omni_disagg_ci seedtts c16 beside text video"
  taskset -c $ALL python3 $K/baseline.py --tree $tree --model omni_disagg_ci --port 28400 --out $O/$name \
    --loads 16 --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:28400/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $ALL python3 $K/videomme_speed.py --host 127.0.0.1 --port 28400 --repo-id zhaochenyang20/Video_MME_ci \
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 4 \
        --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
      client=$!
      while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
      kill $client 2>/dev/null
      wait $client 2>/dev/null
      pass=$((pass + 1))
    done
  ) &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
log "q1115_mainmix start"
for name in md_m1 md_m2; do
  quiet_host
  mixed $name $MAIN
done
log "q1115_mainmix done"
