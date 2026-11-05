#!/usr/bin/env bash
# 11-06, node-0000, after q1106_3451: #3452's new form against upstream main 824d64358, same time, each server on its card's socket, every arm on
# both cards. Speech is on main's exact path, so it should read level. Video should gain. Mixed traffic is what the form is for: speech first
# audio while videos run. Then two-card speech on the full host and on one socket, and the CI's two-card text Video-MME for accuracy.
set -u
O=/data/rp1106
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main L=/workspace/lane
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167 CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
cpus() { if [ "$1" = 0 ]; then echo $CPUS0; else echo $CPUS1; fi; }
port() { if [ "$1" = 0 ]; then echo 28400; else echo 28800; fi; }
until grep -q "q1106_3451 done" $O/queue.log 2>/dev/null; do sleep 15; done
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 load $(tail -1 $O/cpu_load.log) start"; }
speech() {
  local name=$1 tree=$2 card=$3
  note $name $tree "seedtts card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 tree=$2 card=$3 bench=$4
  note $name $tree "$bench card $card"
  taskset -c $(cpus $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $(port $card) --out $O/$name \
    --bench $bench --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
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
        --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1
      pass=$((pass + 1))
    done
  ) &
  local client=$!
  wait $driver
  log "boot $name rc $?"
  wait $client
}
twocard() {
  local name=$1 tree=$2 cpuset=$3
  note $name $tree "two card cpus $cpuset"
  taskset -c $cpuset python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name \
    --loads 64,16 --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
twocard_video() {
  local name=$1 tree=$2
  note $name $tree "two card text video"
  python3 $K/baseline.py --tree $tree --model omni_speech --card 0 --port 28400 --out $O/$name \
    --bench videomme_text --loads 16 --passes clean,events --bench-repeats 2 --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
log "q1106_lane start"
speech ls_m1 $MAIN 0 & speech ls_l1 $L 1 & wait
speech ls_l2 $L 0 & speech ls_m2 $MAIN 1 & wait
speech ls_m3 $MAIN 0 & speech ls_l3 $L 1 & wait
speech ls_l4 $L 0 & speech ls_m4 $MAIN 1 & wait
log "cell ls done"
video lt_m1 $MAIN 0 videomme_text & video lt_l1 $L 1 videomme_text & wait
video lt_l2 $L 0 videomme_text & video lt_m2 $MAIN 1 videomme_text & wait
log "cell lt done"
video lv_m1 $MAIN 0 videomme & video lv_l1 $L 1 videomme & wait
video lv_l2 $L 0 videomme & video lv_m2 $MAIN 1 videomme & wait
log "cell lv done"
mixed lx_m1 $MAIN 0 & mixed lx_l1 $L 1 & wait
mixed lx_l2 $L 0 & mixed lx_m2 $MAIN 1 & wait
log "cell lx done"
ALL=0-223
twocard la_m1 $MAIN $ALL
twocard la_l1 $L $ALL
twocard la_l2 $L $ALL
twocard la_m2 $MAIN $ALL
log "cell la done"
twocard lo_m1 $MAIN $CPUS1
twocard lo_l1 $L $CPUS1
twocard lo_l2 $L $CPUS1
twocard lo_m2 $MAIN $CPUS1
log "cell lo done"
twocard_video lc_m1 $MAIN
twocard_video lc_l1 $L
twocard_video lc_l2 $L
twocard_video lc_m2 $MAIN
log "q1106_lane done"
