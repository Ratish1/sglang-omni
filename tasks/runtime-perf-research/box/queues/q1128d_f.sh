#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_e: speech c16 closed loop beside 8 looping videos, colocated, with the
# thinker's radix cache off, so every video prefills its whole prompt in every pass, as unique videos do. The
# q1128d_b closed loop read its measured pass on prefix hits, where the PR had no underruns, while its first pass
# after boot (uncached) had 37 underrun streams against main's 10. FIX 91c2e2010 against MAIN 0f70d9c7c, same-time
# pairs swapped.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main F=/workspace/fix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
UNCACHED="--thinker.engine.disable_radix_cache true"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
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
until grep -q "q1128d_e done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_f start"
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
closed() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  mkdir -p $O/$name
  note $name $tree "colocated, thinker radix cache off, speech c16 closed loop beside 8 looping videos, card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K $UNCACHED" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port $cpus &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
both "closed cu1_m $M 0 4500" "closed cu1_f $F 1 4500"
both "closed cu2_f $F 0 4700" "closed cu2_m $M 1 4700"
python3 $K/body_table.py $O c16 "the PR against main, speech c16 closed loop beside 8 looping uncached videos, colocated" \
  main=cu1_m,cu2_m fix=cu1_f,cu2_f >> $O/queue.log 2>&1
python3 $K/playback_margin.py $O/cu1_m $O/cu2_m $O/cu1_f $O/cu2_f >> $O/queue.log 2>&1
log "closed uncached read"
log "q1128d_f done"
