#!/usr/bin/env bash
# 11-14, node-0000, after q1114_fix2. Speech c16 beside a looping Video-MME client (4 in flight):
# mx3_f, #3451 with the sync-free build alone (FIX), the landing base #3452's own pair is read on (mx3_f against mx3_x);
# mxr_m, mxr_f, mxr_x, speech at a fixed 9 requests a second (16 in flight at most), so every arm serves the same speech
# load and the video client's latency is read at equal offered load. Main has no fixed-rate cell on file, so it runs once.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main FIX=/workspace/fix STACKFIX=/workspace/stackfix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
until grep -q "q1114_fix2 done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1114_fix3 start"
mixed() {
  local name=$1 tree=$2 extra=$3
  note $name $tree "seedtts c16 beside text video $extra"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events $extra > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:28400/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $CARD_CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port 28400 --repo-id zhaochenyang20/Video_MME_ci \
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
mixed mx3_f $FIX ""
RATE="--bench-extra=--request-rate=9"
mixed mxr_m $MAIN "$RATE"
mixed mxr_f $FIX "$RATE"
mixed mxr_x $STACKFIX "$RATE"
log "q1114_fix3 done"
