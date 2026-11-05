#!/usr/bin/env bash
# 11-06, node-0000, after q1106_twocard. Passive OpenMP costs the serial path: preprocessing work +0.29 ms on speech, +0.1 ms on MMSU where that stage is the limit, req/s about -0.6 %. Earlier forms read
# the two-card gain mostly from the team size. Does the lane hold without the policy? Arm Y is the lane with only that default removed. Against the lane: two-card speech, speech beside video, text video
# with its CPU, the CI's two-card text video. Against main: MMSU on the fused pipeline, where the lean showed.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main L=/workspace/lane Y=/workspace/lane_spin
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_twocard done" $O/queue.log 2>/dev/null; do sleep 15; done
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 load $(tail -1 $O/cpu_load.log) start"; }
twocard() {
  local name=$1 tree=$2
  note $name $tree "two card"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
twocard_video() {
  local name=$1 tree=$2
  note $name $tree "two card text video"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 tree=$2 card=$3
  note $name $tree "colocated text video card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
mmsu() {
  local name=$1 tree=$2 card=$3
  note $name $tree "fused mmsu card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_text_fused --card $card --port $(port $card) --out $O/$name \
    --bench mmsu --loads 96,16 --passes warm,clean,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
mixed() {
  local name=$1 tree=$2 card=$3
  note $name $tree "seedtts beside text video card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$(port $card)/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $(cpus $card) python3 $K/videomme_speed.py --host 127.0.0.1 --port $(port $card) --repo-id zhaochenyang20/Video_MME_ci \
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
log "q1106_nopassive start"
twocard ya_l1 $L
twocard ya_y1 $Y
twocard ya_y2 $Y
twocard ya_l2 $L
log "cell ya done"
mixed yx_l1 $L 0 & mixed yx_y1 $Y 1 & wait
mixed yx_y2 $Y 0 & mixed yx_l2 $L 1 & wait
log "cell yx done"
video yt_l1 $L 0 & video yt_y1 $Y 1 & wait
video yt_y2 $Y 0 & video yt_l2 $L 1 & wait
log "cell yt done"
mmsu yf_m1 $MAIN 0 & mmsu yf_y1 $Y 1 & wait
mmsu yf_y2 $Y 0 & mmsu yf_m2 $MAIN 1 & wait
log "cell yf done"
twocard_video yc_l1 $L
twocard_video yc_y1 $Y
twocard_video yc_y2 $Y
twocard_video yc_l2 $L
log "q1106_nopassive done"
