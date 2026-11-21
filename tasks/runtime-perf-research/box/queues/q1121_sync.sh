#!/usr/bin/env bash
# 11-21, node-0001, card 1 while card 0 measures, before q1121_final2's WER. q1121_tw2 showed thinker turn holds of 200
# to 218 ms on steps of about 6,000 extend tokens, inside the step before finalize's release: a host wait for the
# forward's device time. Candidate: post_prefill moves the speech rows' offsets with a blocking H2D copy
# (thinker_model_runner.py:151), which waits for the stream. Check boot (OCVG 85ba60a52, speech 8/s beside video 1/s)
# with torch's sync debug mode counting every synchronizing CUDA call by thread and site.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
OCVGP=/workspace/ocvg_prof
OCVG_SHA=85ba60a52
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=56-111,168-223
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
video_loop() {
  local name=$1 tree=$2 port=$3 driver=$4
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 --request-rate 1 \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
}
log "q1121_sync start"
if [ "$(git -C $OCVGP rev-parse --short=9 HEAD)" != "$OCVG_SHA" ] || ! git -C $OCVGP diff --quiet HEAD -- sglang_omni; then
  log "q1121_sync stopped: ocvg_prof is not a clean $OCVG_SHA"
  exit 1
else :; fi
name=sy1_b port=31610
note $name $OCVGP "seedtts 8/s beside video 1/s card 1 sync site probe port $port"
taskset -c $CPUS python3 $K/baseline.py --tree $OCVGP --model omni_colocated --card 1 --port $port --out $O/$name \
  --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
  --probe --serve-env "OMNI_HOP_PROBE_SYNC_SITES=1" > $O/$name.driver.log 2>&1 &
driver=$!
until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
video_loop $name $OCVGP $port $driver &
looper=$!
wait $driver
log "boot $name rc $?"
wait $looper
for cell in c64_warm c64_events; do
  python3 $K/sync_window.py $O/$name --cell $cell --thread scheduler > $O/$name.sync_$cell.txt 2>&1
done
log "SY read"
log "q1121_sync done"
