#!/usr/bin/env bash
# 11-21, node-0001, after q1121_swap: the question itself, the stack with both changes (NSBOTH 054cea323) against main
# (NMAIN e0482a08c), colocated, same-time pairs on both cards, swapped, probes on: speech 8/s seed 3451 beside video
# 1/s seed 7 (cap 8), then the 11-20 cell where #3452's inter chunk cost was read, speech 8/s beside the looping
# Video-MME client (4 in flight).
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain NSBOTH=/workspace/nsboth
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
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
  local name=$1 tree=$2 card=$3 offset=$4 video=$5
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  local video_args=(--max-concurrency 4)
  if [ $video = rate ]; then video_args=(--max-concurrency 8 --request-rate 1); else :; fi
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video $video card $card probes"
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
        --max-tokens 256 --timeout-s 500 --warmup 0 "${video_args[@]}" \
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
until grep -q "q1121_swap done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_final start"
quiet_host
both "mixed l1_b0 $NSBOTH 0 200 rate" "mixed l1_m1 $NMAIN 1 200 rate"
quiet_host
both "mixed l2_m0 $NMAIN 0 400 rate" "mixed l2_b1 $NSBOTH 1 400 rate"
python3 $K/body_table.py $O c64 "stack with both changes against main, speech 8/s and video 1/s" main=l1_m1,l2_m0 \
  stack_both=l1_b0,l2_b1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O l1_b0 l2_b1 l1_m1 l2_m0 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O main=l1_m1,l2_m0 stack_both=l1_b0,l2_b1 > $O/final_first_gap_rate.txt 2>&1
log "final rate read"
quiet_host
both "mixed m1_b0 $NSBOTH 0 600 loop" "mixed m1_m1 $NMAIN 1 600 loop"
quiet_host
both "mixed m2_m0 $NMAIN 0 800 loop" "mixed m2_b1 $NSBOTH 1 800 loop"
python3 $K/body_table.py $O c64 "stack with both changes against main, 8/s beside 4 looping videos" main=m1_m1,m2_m0 \
  stack_both=m1_b0,m2_b1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O m1_b0 m2_b1 m1_m1 m2_m0 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O main=m1_m1,m2_m0 stack_both=m1_b0,m2_b1 > $O/final_first_gap_loop.txt 2>&1
log "q1121_final done"
