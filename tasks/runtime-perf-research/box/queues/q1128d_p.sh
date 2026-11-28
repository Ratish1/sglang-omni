#!/usr/bin/env bash
# 11-28 second lease, node-0001: what a talker row waits for when it sits out a decode step. The traced pair (q1128d_l)
# showed long chunk gaps in which the talker keeps stepping but leaves the stream out of about 41 of 52 steps, with its
# thinker text already sent and nothing waiting in the talker inbox. The hop probe now records, per talker batch update,
# each row not ready and its wait (its feedback row, its next text, the pad after the thinker's last text). H20
# emulation, speech c16 closed loop beside 8 looping videos, MAIN against the PR head 818e48b43, same time.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main D4=/workspace/dev4
H20_FREE_MIB=97871
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_HOP_PROBE_TALKER_BATCHES=1
export OMNI_HOP_PROBE_THINKER_STEPS=1
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
video_loop() {
  local name=$1 tree=$2 driver=$3 port=$4 cpus=$5
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
}
traced() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  mkdir -p $O/$name
  python3 $K/ballast.py --card $card --free-mib $H20_FREE_MIB --ready-file $O/$name/ballast.txt > $O/$name/ballast.log 2>&1 &
  local ballast=$!
  until [ -s $O/$name/ballast.txt ] || ! kill -0 $ballast 2>/dev/null; do sleep 2; done
  note $name $tree "H20 colocated bytes on card $card, hop probe with row waits, speech c16 closed loop beside 8 looping videos"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated_h20_on_h200 --card $card --port $port \
    --out $O/$name --loads 16 --omni-chunk-audio --probe --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port $cpus &
  local looper=$!
  wait $driver
  local rc=$?
  wait $looper
  kill $ballast 2>/dev/null
  wait $ballast 2>/dev/null
  log "boot $name rc $rc row wait records $(cat $O/$name/probe/*.jsonl 2>/dev/null | grep -c '\"k\": \"te\"')"
}
log "q1128d_p start"
both "traced hp_m $M 0 2200" "traced hp_v $D4 1 2200"
python3 $K/playback_margin.py $O/hp_m $O/hp_v >> $O/queue.log 2>&1
python3 $K/long_gaps.py --cell c16_events --min-ms 600 $O/hp_m $O/hp_v > $O/hp_long_gaps.txt 2>&1
log "row waits read"
log "q1128d_p done"
