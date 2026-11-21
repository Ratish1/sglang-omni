#!/usr/bin/env bash
# 11-21, node-0001, after q1121_v2. The graph change alone against main (E4) costs the first window 8.4 and 10.7 ms in
# both same-time pairs: the talker's median frame interval 12.2 against 10.7 ms, code2wav ingest +1 ms, while the
# thinker's mean step interval drops 3 %. Thinker and talker are separate processes on one card without MPS, so they
# time-slice; an eager video admission step leaves its card mostly idle for about 70 ms a second. Settle the mechanism:
#   N1/N2  main and the graph change, same time on both cards then swapped, speech 8/s beside video 1/s, a 15 s CUDA
#          trace (graph level, no context switch collection) 20 s into the measured pass; talker steps (wall, own and
#          other processes' device time inside) and the thinker's eager steps per arm
#   P      the graph change on its other landing base: both changes (PBOTH2) against the receive-only change (PSHM2)
#   A      main against main at the same time, the card bias of a same-time pair in this cell
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain PFIX=/workspace/pfix PSHM2=/workspace/pshm2 PBOTH2=/workspace/pboth2
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
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
video_loop() {
  local name=$1 tree=$2 cpus=$3 port=$4 driver=$5
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 --request-rate 1 \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
}
mixed() {
  local name=$1 tree=$2 card=$3 offset=$4 kind=$5
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  local pass_args=(--passes warm,events)
  if [ $kind = nsys ]; then
    pass_args=(--passes warm,nsys --nsys-session $name --nsys-trace cuda --nsys-graph-trace graph --nsys-delay 20
      --nsys-seconds 15)
  else :; fi
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s seed 7 cap 8 card $card $kind"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" "${pass_args[@]}" \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $cpus $port $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
  if [ $kind = nsys ]; then
    (cd $O/$name && python3 - c64_nsys/trace.sqlite serve.log < $K/talker_steps.py > talker_steps.txt 2>&1
      python3 - c64_nsys/trace.sqlite serve.log < $K/thinker_steps.py > thinker_steps.txt 2>&1)
    log "trace read $name $(grep -c . $O/$name/talker_steps.txt) talker lines, $(grep -c . $O/$name/thinker_steps.txt) thinker lines"
  else :; fi
}
pair_read() {
  local title=$1 cell=$2 base=$3 change=$4
  python3 $K/body_table.py $O $cell "$title" "$base" "$change" >> $O/queue.log 2>&1
}
until grep -q "q1121_v2 done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1121_v3 start"
quiet_host
both "mixed n1_m1 $NMAIN 1 150 nsys" "mixed n1_p0 $PFIX 0 150 nsys"
quiet_host
both "mixed n2_m0 $NMAIN 0 350 nsys" "mixed n2_p1 $PFIX 1 350 nsys"
log "N read"
quiet_host
both "mixed p1_b0 $PBOTH2 0 550 events" "mixed p1_x1 $PSHM2 1 550 events"
quiet_host
both "mixed p2_x0 $PSHM2 0 750 events" "mixed p2_b1 $PBOTH2 1 750 events"
pair_read "graph change on the receive-only base, speech 8/s and video 1/s" c64 shm=p1_x1,p2_x0 both=p1_b0,p2_b1
log "P read"
quiet_host
both "mixed a1_m0 $NMAIN 0 950 events" "mixed a1_m1 $NMAIN 1 950 events"
pair_read "main card 0 against main card 1, same time" c64 card1=a1_m1 card0=a1_m0
log "q1121_v3 done"
