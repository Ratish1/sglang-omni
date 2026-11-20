#!/usr/bin/env bash
# 11-20, node-0000, after q1120_c64b: speech at 8/s on one seeded arrival schedule (cap 64) beside the looping Video-MME
# client (4 in flight), colocated, probed, same-time pairs on both cards, swapped: the stack (STACK 6149c714f) against
# the video lane alone on the same main (V3452 a18f40fa7), then the stack against main (de4cb9f0f). The closed c16 loop
# beside 8 videos holds 16 speech requests in flight, so time the talker gives back moves into the thinker phase where
# first audio is counted; equal speech arrivals read each request's first audio and chunk cadence at the same offered
# work.
set -u
O=/data/v1120/slot2
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/tmain V3452=/workspace/v3452 STACK=/workspace/tstack
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
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
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside text video card $card probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=8" \
    --probe --serve-env OMNI_HOP_PROBE_TALKER_BATCHES=1 > $O/$name.driver.log 2>&1 &
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
until grep -q "q1120_c64b done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1120_mixed_eq start"
quiet_host
both "mixed e1_t0 $STACK 0 2300" "mixed e1_v1 $V3452 1 2300"
quiet_host
both "mixed e2_v0 $V3452 0 2500" "mixed e2_t1 $STACK 1 2500"
quiet_host
both "mixed e3_t0 $STACK 0 2700" "mixed e3_m1 $MAIN 1 2700"
quiet_host
both "mixed e4_m0 $MAIN 0 2900" "mixed e4_t1 $STACK 1 2900"
python3 $K/body_table.py $O c64 "stack against the video lane alone, 8/s beside video" lane=e1_v1,e2_v0 stack=e1_t0,e2_t1 \
  >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "stack against main, 8/s beside video" main=e3_m1,e4_m0 stack=e3_t0,e4_t1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O e1_t0 e2_t1 e1_v1 e2_v0 e3_t0 e4_t1 e3_m1 e4_m0 >> $O/queue.log 2>&1
python3 $K/chunk_gap_parts.py --cell c64_events $O/e1_t0 $O/e2_t1 $O/e1_v1 $O/e2_v0 $O/e3_m1 $O/e4_m0 \
  > $O/chunk_gap_parts_mixed_eq.txt 2>&1
log "q1120_mixed_eq done"
