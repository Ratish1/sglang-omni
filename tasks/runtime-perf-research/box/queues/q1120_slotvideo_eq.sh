#!/usr/bin/env bash
# 11-20, node-0000: speech at 6/s on one seeded arrival schedule (cap 64) beside the looping Video-MME client (4 in
# flight), colocated, probed, the talker's slot state alone (SLOT d3806fb6a) against main (de4cb9f0f), same-time pairs on
# both cards, swapped. In the closed c16 cell beside 8 videos the slot arm's first audio rose 4.1 %, 52 of its 56 ms in
# main's serial preprocessing queue; 6/s stays below what main serves beside video (7.6 to 7.7 at 8/s), so both arms
# keep up and each request's first audio is read at the same offered work.
set -u
O=/data/v1120/slot2
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/tmain SLOT=/workspace/tslot
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
  note $name $tree "seedtts 6/s seed $OMNI_ARRIVAL_SEED cap 64 beside text video card $card probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=6" \
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
log "q1120_slotvideo_eq start"
quiet_host
both "mixed f1_s0 $SLOT 0 3500" "mixed f1_m1 $MAIN 1 3500"
quiet_host
both "mixed f2_m0 $MAIN 0 3700" "mixed f2_s1 $SLOT 1 3700"
python3 $K/body_table.py $O c64 "slot against main, 6/s beside video" main=f1_m1,f2_m0 slot=f1_s0,f2_s1 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O f1_s0 f2_s1 f1_m1 f2_m0 >> $O/queue.log 2>&1
python3 $K/first_audio_states.py --cell c64_events --main $O/f1_m1,$O/f2_m0 --branch $O/f1_s0,$O/f2_s1 \
  > $O/first_audio_states_slotvideo_eq.txt 2>&1
log "q1120_slotvideo_eq done"
