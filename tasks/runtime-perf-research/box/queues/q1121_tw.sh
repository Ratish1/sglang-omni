#!/usr/bin/env bash
# 11-21, node-0001, card 1 while card 0 measures, before q1121_final2's WER. How long the talker waits for the turn in
# one context and what the thinker was doing: the combined form (OCVG 85ba60a52) at speech 8/s beside video 1/s, warm
# pass (every video new) and events pass, with the turn probe only (OMNI_HOP_PROBE_TURNS=1).
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
log "q1121_tw start"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-one-context-visual-graph
git worktree add -q --detach $OCVGP FETCH_HEAD
if [ "$(git -C $OCVGP rev-parse --short=9 HEAD)" != "$OCVG_SHA" ] || ! git -C $OCVGP diff --quiet HEAD -- sglang_omni; then
  log "q1121_tw stopped: ocvg_prof is not a clean $OCVG_SHA"
  exit 1
else :; fi
name=tw1_b port=31690
note $name $OCVGP "seedtts 8/s beside video 1/s card 1 turn probe port $port"
taskset -c $CPUS python3 $K/baseline.py --tree $OCVGP --model omni_colocated --card 1 --port $port --out $O/$name \
  --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
  --probe --serve-env "OMNI_HOP_PROBE_TURNS=1" > $O/$name.driver.log 2>&1 &
driver=$!
until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
video_loop $name $OCVGP $port $driver &
looper=$!
wait $driver
log "boot $name rc $?"
wait $looper
for cell in c64_warm c64_events; do
  python3 $K/turn_waits.py $O/$name --cell $cell > $O/$name.turns_$cell.txt 2>&1
done
grep -h "waiter priority" $O/$name.turns_c64_*.txt >> $O/queue.log
log "TW read"
log "q1121_tw done"
