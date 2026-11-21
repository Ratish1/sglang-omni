#!/usr/bin/env bash
# 11-21, node-0001, card 1 beside q1121_v7 on card 0. Design input for placing the thinker and the talker (with
# code2wav) in one process: what share of each second every server process and thread holds the interpreter lock at
# speech 8/s beside video 1/s, so the lock budget of one process holding both schedulers is measured, not assumed.
# py-spy --gil (one sample per tick, the lock holder) on every server process for 15 s, 20 s into the pass; then an
# events pass. A budget, not a comparison: card 1's NUMA node is short of free pages (video preprocessing slower), the
# thinker's and talker's Python is not page bound.
#   G1  main
#   G2  the stack with both changes
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain NSBOTH2=/workspace/nsboth2
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=56-111,168-223
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
video_loop() {
  local name=$1 tree=$2 port=$3 driver=$4
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
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
gil() {
  local name=$1 tree=$2 port=$3
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED beside video 1/s card 1 pyspygil then events port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" \
    --passes warm,pyspygil,events --pyspy-delay 20 --pyspy-seconds 15 > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $port $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
  log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name), $(node_free)"
}
log "q1121_g1 start, $(node_free)"
gil g1_m $NMAIN 31190
gil g2_s $NSBOTH2 31390
log "q1121_g1 done"
