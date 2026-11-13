#!/usr/bin/env bash
# 11-13, node-0000, one H200: replaces q1113_gpu after its id_m boot, to put a sync call site probe ahead of the A/B cells. The traces
# showed the thinker's host blocked 35 to 42 % of the window in device-to-host copies of 64 bytes or less into pageable memory; the probe
# boot (main, c16) records the Python stack of every synchronizing CUDA operation in each GPU process. Then the cells q1113_gpu had left.
set -u
O=/data/rp1113
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main P3S=/workspace/p3s LANE=/workspace/lane WAIT=/workspace/lane_wait
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CARD_CPUS=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 load $(tail -1 $O/cpu_load.log) start"; }
while pgrep -f "baseline.py --tree" > /dev/null; do sleep 5; done
log "q1113_gpu2 start, id_m finished under the replaced queue"

identity() {
  local name=$1 tree=$2
  note $name $tree "seeded c1"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 1 --passes clean "--bench-extra=--seed 3451" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
identity id_s $P3S
log "cell C1 done"

probe() {
  local name=$1 tree=$2
  note $name $tree "sync probe c16"
  mkdir -p $O/$name
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,events --serve-env "PYTHONPATH=$K/sync_probe:$tree OMNI_SYNC_PROBE=$O/$name/sync" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
probe sp_m $MAIN
log "cell P done"

speech16() {
  local name=$1 tree=$2
  note $name $tree "c16"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech16 s16_m1 $MAIN
speech16 s16_s1 $P3S
speech16 s16_s2 $P3S
speech16 s16_m2 $MAIN
log "cell C3 done"

mixed() {
  local name=$1 tree=$2
  note $name $tree "seedtts beside text video"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1 &
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
mixed mx_m1 $MAIN
mixed mx_w1 $WAIT
mixed mx_p1 $LANE
mixed mx_p2 $LANE
mixed mx_w2 $WAIT
mixed mx_m2 $MAIN
log "cell B1 done"

mmsu() {
  local name=$1 tree=$2
  note $name $tree "fused mmsu"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_text_fused --card 0 --port 28400 --out $O/$name \
    --bench mmsu --loads 96,16 --passes warm,clean,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
mmsu mu_m1 $MAIN
mmsu mu_w1 $WAIT
mmsu mu_w2 $WAIT
mmsu mu_m2 $MAIN
log "cell B2 done"

speech64() {
  local name=$1 tree=$2
  note $name $tree "c64"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --loads 64 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech64 s64_m1 $MAIN
speech64 s64_s1 $P3S
speech64 s64_s2 $P3S
speech64 s64_m2 $MAIN
log "cell C4 done"

video() {
  local name=$1 tree=$2
  note $name $tree "colocated text video"
  taskset -c $CARD_CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video vt_w1 $WAIT
video vt_p1 $LANE
video vt_p2 $LANE
video vt_w2 $WAIT
log "q1113_gpu2 done"
