#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_j: speech on two GPUs (Qwen3OmniSpeechPipelineConfig's default placement,
# thinker and encoders on GPU 0, talker and code2wav on GPU 1), the one speech layout with no reading. DEV2 (4cac082f9)
# against MAIN 0f70d9c7c, boots alternating M V V M (each server takes both cards): speech 8/s open loop (seed 3451,
# cap 64) beside video 1/s (seed 7, cap 8), warm and events passes, first with the thinker's prefix cache on, then off.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main V=/workspace/dev2
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
UNCACHED="--thinker.engine.disable_radix_cache true"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
video_loop() {
  local name=$1 tree=$2 driver=$3 port=$4
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
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
equal_load() {
  local name=$1 tree=$2 port=$3 extra=$4
  mkdir -p $O/$name
  note $name $tree "speech on two GPUs, speech 8/s beside video 1/s, ${extra:-prefix cache on}"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K $extra" --omni-chunk-audio "--bench-extra=--request-rate=8" \
    --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
until grep -q "q1128d_j done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_k start"
equal_load tg1_m $M 21000 ""
equal_load tg2_v $V 21200 ""
equal_load tg3_v $V 21400 ""
equal_load tg4_m $M 21600 ""
python3 $K/body_table.py $O c64 "dev2 against main, speech on two GPUs, speech 8/s beside video 1/s, prefix cache on" \
  main=tg1_m,tg4_m fix=tg2_v,tg3_v >> $O/queue.log 2>&1
python3 $K/playback_margin.py $O/tg1_m $O/tg4_m $O/tg2_v $O/tg3_v >> $O/queue.log 2>&1
log "two gpu cached read"
equal_load tu1_m $M 21800 "$UNCACHED"
equal_load tu2_v $V 22000 "$UNCACHED"
equal_load tu3_v $V 22200 "$UNCACHED"
equal_load tu4_m $M 22400 "$UNCACHED"
python3 $K/body_table.py $O c64 "dev2 against main, speech on two GPUs, speech 8/s beside video 1/s, prefix cache off" \
  main=tu1_m,tu4_m fix=tu2_v,tu3_v >> $O/queue.log 2>&1
python3 $K/playback_margin.py $O/tu1_m $O/tu4_m $O/tu2_v $O/tu3_v >> $O/queue.log 2>&1
log "two gpu uncached read"
log "q1128d_k done"
