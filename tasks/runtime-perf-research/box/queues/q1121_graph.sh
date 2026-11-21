#!/usr/bin/env bash
# 11-21, node-0001, after q1121_parts: whether a thinker mixed step that holds a long-context row (a video request's
# answer decoding) replays a prefill graph or runs eager, and what it costs by the rows' context. In pp1 a speech
# admission's mixed step costs 12 to 14 ms with only speech rows and 25 to 34 ms with a video request's row in it, while
# the same row adds 3.5 ms to a decode step. The parts probe now records each step's graph replay, its longest row
# context and its rows over 4,096 tokens. Stack (STACK 6149c714f) card 1 against main (de4cb9f0f) card 0, speech at 8/s
# on one seeded arrival schedule beside 4 looping videos.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/tmain STACK=/workspace/tstack
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_THINKER_PARTS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((28000 + 2000 * $1 + $2)); }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
mixed() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside 4 videos card $card parts graph"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=8" \
    --probe --serve-env "$PROBES" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
        --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
        --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 4 \
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
until grep -q "q1121_parts done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_graph start"
quiet_host
both "mixed pg1_m0 $MAIN 0 2100" "mixed pg1_s1 $STACK 1 2100"
python3 $K/thinker_parts.py --cell c64_events $O/pg1_m0 $O/pg1_s1 > $O/thinker_parts_graph.txt 2>&1
log "q1121_graph done"
