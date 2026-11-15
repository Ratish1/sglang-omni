#!/usr/bin/env bash
# 11-15, node-0000: the swapped video pair again (v2 was void, host load 187 to 356 from another tenant), FIX on card 0
# and FOLDSYNC on card 1, speech 8/s seed 3451 cap 64 beside the looping Video-MME client. A boot starts only once the
# host's one-minute load is under 80.
set -u
O=/data/rp1115
K=/workspace/kit2/tasks/runtime-perf-research/box
FIX=/workspace/fix FOLDSYNC=/workspace/foldsync
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
CPUS0=0-55,112-167
CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
log "q1115_mixed2 start"
mixed() {
  local name=$1 tree=$2 card=$3
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside text video card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=8" \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port --repo-id zhaochenyang20/Video_MME_ci \
        --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 4 \
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
quiet_host
mixed v3_x0 $FIX 0 &
first=$!
mixed v3_s1 $FOLDSYNC 1 &
second=$!
wait $first $second
python3 $K/mixed_cell_load.py $O v1_s0 v3_s1 v1_x1 v3_x0 >> $O/queue.log 2>&1
log "q1115_mixed2 done"
