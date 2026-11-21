#!/usr/bin/env bash
# 11-21, node-0001, card 1 beside q1121_v7 on card 0. Smoke boot of the one-context prototype (ONECTX 76ddf8359, tree
# 409b390b602b, on both changes): thinker, talker and code2wav in one process, a turn per engine, the talker's loop on
# the top priority stream and the thinker's on the lowest. Speech 8/s seed 3451 beside video 1/s seed 7 (cap 8), warm
# and events passes. Card 1's NUMA node is short of free pages, so its numbers are not compared; it settles whether the
# process builds both engines and serves.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
ONECTX=/workspace/onectx
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=56-111,168-223
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
name=oc1_x1
port=31590
log "boot $name $(git -C $ONECTX rev-parse --short HEAD) $(git -C $ONECTX rev-parse --short=12 HEAD:sglang_omni) one-context smoke card 1 port $port start, $(node_free)"
taskset -c $CPUS python3 $K/baseline.py --tree $ONECTX --model omni_colocated --card 1 --port $port --out $O/$name \
  --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
  > $O/$name.driver.log 2>&1 &
driver=$!
until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
log "oc1 health $(curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null && echo up || echo down)"
(
  cd $ONECTX
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$ONECTX taskset -c $CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 --request-rate 1 \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
) &
looper=$!
wait $driver
log "boot $name rc $?"
wait $looper
log "q1121_oc done"
