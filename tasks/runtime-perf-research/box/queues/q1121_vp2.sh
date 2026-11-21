#!/usr/bin/env bash
# 11-21, node-0001, card 1 beside the card 0 queues. Part 1 at a377a4ad0 (tree c70c67ce9eb5) against the graph change
# alone (PFIX 09dae918c): speech 8/s beside video 1/s, warm pass (every video new) and events pass, with the thinker
# parts probe, so each synchronous thinker execute records whether it replayed a graph, its host launch ms and its
# device span. Part 1's claim: chunks that compute visual rows replay the prefill graph and launch in about a
# millisecond of host time instead of an eager forward's tens.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
PFIX=/workspace/pfix PVG2=/workspace/pvg2
PVG2_SHA=a377a4ad0
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_THINKER_PARTS=1"
CPUS=56-111,168-223
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
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
parts_mixed() {
  local name=$1 tree=$2 port=$3
  note $name $tree "seedtts 8/s beside video 1/s card 1 parts probed port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
    --probe --serve-env "$PROBES" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $port $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
log "q1121_vp2 start, $(node_free)"
if [ "$(git -C $PVG2 rev-parse --short=9 HEAD)" != "$PVG2_SHA" ] || ! git -C $PVG2 diff --quiet HEAD -- sglang_omni; then
  log "q1121_vp2 stopped: pvg2 is not a clean $PVG2_SHA"
  exit 1
else :; fi
parts_mixed vx1_g2 $PVG2 31870
parts_mixed vx2_f1 $PFIX 31830
for boot in vx1_g2 vx2_f1; do
  for cell in c64_warm c64_events; do
    python3 $K/thinker_parts.py --cell $cell $O/$boot > $O/$boot.parts_$cell.txt 2>&1
  done
done
python3 $K/long_thinker_steps.py $O vx1_g2 vx2_f1 >> $O/queue.log 2>&1
log "VX read"
log "q1121_vp2 done"
