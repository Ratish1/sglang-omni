#!/usr/bin/env bash
# 11-21, node-0001, after q1121_profile: what the thinker's prefill chunk size alone does beside video, a what-if through
# the existing engine override, not a candidate. The stack (STACK 6149c714f) with chunked_prefill_size 2048 and 4096
# against its default 8192, speech at 8/s on one seeded arrival schedule (cap 64) beside the looping Video-MME client (4
# in flight), colocated, both probes on, same-time pairs on both cards, swapped.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
STACK=/workspace/tstack
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1"
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
  local name=$1 tree=$2 card=$3 offset=$4 chunk=$5
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  local extra="$CONTEXT_32K"
  if [ $chunk != default ]; then extra="$extra --thinker.engine.chunked_prefill_size $chunk"; else :; fi
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside 4 videos card $card chunk $chunk probes"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$extra" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=8" \
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
until grep -q "q1121_profile done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_chunk start"
quiet_host
both "mixed ck1_a0 $STACK 0 900 2048" "mixed ck1_d1 $STACK 1 900 default"
quiet_host
both "mixed ck2_d0 $STACK 0 1100 default" "mixed ck2_a1 $STACK 1 1100 2048"
quiet_host
both "mixed ck3_b0 $STACK 0 1300 4096" "mixed ck3_d1 $STACK 1 1300 default"
quiet_host
both "mixed ck4_d0 $STACK 0 1500 default" "mixed ck4_b1 $STACK 1 1500 4096"
python3 $K/body_table.py $O c64 "stack chunk 2048 against 8192, 8/s beside video" d8192=ck1_d1,ck2_d0 c2048=ck1_a0,ck2_a1 \
  >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "stack chunk 4096 against 8192, 8/s beside video" d8192=ck3_d1,ck4_d0 c4096=ck3_b0,ck4_b1 \
  >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O ck1_a0 ck2_a1 ck1_d1 ck2_d0 ck3_b0 ck4_b1 ck3_d1 ck4_d0 >> $O/queue.log 2>&1
python3 $K/thinker_cadence.py --cell c64_events $O/ck1_d1 $O/ck2_d0 $O/ck1_a0 $O/ck2_a1 $O/ck3_d1 $O/ck4_d0 $O/ck3_b0 \
  $O/ck4_b1 > $O/thinker_cadence_chunk.txt 2>&1
log "q1121_chunk done"
