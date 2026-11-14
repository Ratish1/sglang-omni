#!/usr/bin/env bash
# 11-14, node-0000: speech at 8 requests a second on one seeded arrival schedule (OMNI_ARRIVAL_SEED) with 64 in flight
# allowed, beside the looping Video-MME client. The 9/s cells capped speech at 16 in flight, so both arms queued at the
# client and each served its own rate; here main (eq_m) and the #3451 candidate FIX (eq_f) get the same arrivals.
set -u
O=/data/rp1114
K=/workspace/kit2/tasks/runtime-perf-research/box
MAIN=/workspace/main FIX=/workspace/fix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
log "q1114_eqload start"
mixed() {
  local name=$1 tree=$2
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside text video"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events "--bench-extra=--request-rate=8" \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:28400/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $CARD_CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port 28400 --repo-id zhaochenyang20/Video_MME_ci \
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
mixed eq_m $MAIN
mixed eq_f $FIX
python3 $K/mixed_cell_load.py $O mxr_m mxr_f eq_m eq_f >> $O/queue.log 2>&1
log "q1114_eqload done"
