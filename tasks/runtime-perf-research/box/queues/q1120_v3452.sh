#!/usr/bin/env bash
# 11-20, node-0000, after q1120_slot_openloop: the preprocessing video lane alone on the same main (V3452 a18f40fa7,
# tree 3f5b31005256) against the stack with the talker's slot state (STACK 6149c714f, tree 0e5588dfb14c), SeedTTS c16
# beside 8 looping videos on the colocated config, probed, same-time pairs on both cards, swapped. Isolates what the
# talker change does under video load.
set -u
O=/data/v1120/slot2
K=/workspace/kit/tasks/runtime-perf-research/box
V3452=/workspace/v3452 STACK=/workspace/tstack
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
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
  note $name $tree "omni_colocated seedtts c16 beside 8 videos card $card probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$CONTEXT_32K" --probe --serve-env OMNI_HOP_PROBE_TALKER_BATCHES=1 \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
        --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
        --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 \
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
until grep -q "q1120_slot_openloop done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1120_v3452 start"
quiet_host
both "mixed cv3_v0 $V3452 0 150" "mixed cv3_t1 $STACK 1 150"
quiet_host
both "mixed cv4_t0 $STACK 0 370" "mixed cv4_v1 $V3452 1 370"
python3 $K/body_table.py $O c16 "stack against the video lane alone, colocated beside video" lane=cv3_v0,cv4_v1 \
  stack=cv3_t1,cv4_t0 >> $O/queue.log 2>&1
python3 $K/chunk_gap_parts.py --cell c16_events $O/cv3_v0 $O/cv4_v1 $O/cv3_t1 $O/cv4_t0 > $O/chunk_gap_parts_v3452.txt 2>&1
log "q1120_v3452 done"
