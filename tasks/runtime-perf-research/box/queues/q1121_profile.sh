#!/usr/bin/env bash
# 11-21, node-0001, after setup_1121: where the speech requests' thinker token gaps go beside video. The stack (STACK
# 6149c714f, main de4cb9f0f plus the talker slot state and #3452) against main, colocated, both probes on (talker
# batches, thinker steps), same-time pairs on both cards, swapped: speech at 8/s on one seeded arrival schedule (cap 64)
# beside the looping Video-MME client (4 in flight), then closed c16 beside 8 videos.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/tmain STACK=/workspace/tstack
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
  local name=$1 tree=$2 card=$3 offset=$4 loads=$5 videos=$6 bench=$7
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "seedtts loads $loads $bench beside $videos videos card $card probes"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events $bench \
    --probe --serve-env "$PROBES" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
        --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
        --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency $videos \
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
until grep -q "setup done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_profile start"
EQ="--bench-extra=--request-rate=8"
quiet_host
both "mixed pe1_t0 $STACK 0 300 64 4 $EQ" "mixed pe1_m1 $MAIN 1 300 64 4 $EQ"
quiet_host
both "mixed pe2_m0 $MAIN 0 500 64 4 $EQ" "mixed pe2_t1 $STACK 1 500 64 4 $EQ"
python3 $K/body_table.py $O c64 "stack against main, 8/s beside video" main=pe1_m1,pe2_m0 stack=pe1_t0,pe2_t1 \
  >> $O/queue.log 2>&1
python3 $K/thinker_cadence.py --cell c64_events $O/pe1_m1 $O/pe2_m0 $O/pe1_t0 $O/pe2_t1 > $O/thinker_cadence_eq.txt 2>&1
log "eq cell read"
quiet_host
both "mixed pc1_t0 $STACK 0 700 16 8 ''" "mixed pc1_m1 $MAIN 1 700 16 8 ''"
python3 $K/body_table.py $O c16 "stack against main, closed c16 beside 8 videos" main=pc1_m1 stack=pc1_t0 >> $O/queue.log 2>&1
python3 $K/thinker_cadence.py --cell c16_events $O/pc1_m1 $O/pc1_t0 > $O/thinker_cadence_closed.txt 2>&1
log "q1121_profile done"
