#!/usr/bin/env bash
# 11-15, node-0000, both H200s: #3452 with each video request taking the stage's video CPUs divided by the videos running
# (SHARE 599c63d24), the fix for the two-card c1 cost. Units on the touched suites; the stage alone by video threads at c1,
# c4, c16 (main's serial path rerun beside it); preprocessing bytes against main at OMP 112; the CI two-card Video-MME
# server (boots tc_s1, tc_s2 beside today's tc_m1, tc_m2 main and tc_l1, tc_l2 previous head); colocated SeedTTS c16
# beside a looping 4-in-flight text Video-MME client, the previous head LANE (ecfebb1f3) against SHARE as same-time pairs,
# one card each on its socket, swapped.
set -u
O=/data/v1115
K=/workspace/kit2/tasks/runtime-perf-research/box
SHARE=/workspace/vshare LANE=/workspace/vlane MAIN=/workspace/main
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
CPUS1=56-111,168-223
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
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
log "q1115_share start"

(cd $SHARE && python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni tests/unit_test/preprocessing \
  tests/unit_test/utils tests/unit_test/minicpm_o tests/unit_test/ming_omni 2>&1 | tail -15) > $O/share_units.log
log "units $(tail -1 $O/share_units.log)"

quiet_host
(cd $SHARE && PYTHONPATH=$SHARE taskset -c $CPUS0 python3 $K/preprocess_team_profile.py \
  --configs 1:112:16:1,8:14:0:1,1:112:16:4,8:14:0:4,8:14:0:16,1:112:16:1,8:14:0:1,8:14:0:16 \
  --out $O/share_profile.json 2>&1 | grep -E "threads|Error|Traceback") > $O/share_profile.log
log "profile done"
(cd $SHARE && PYTHONPATH=$SHARE OMP_NUM_THREADS=14 taskset -c $CPUS0 python3 $K/preprocess_shared_identity.py --threads 8 \
  --image $O/form_image.png --out $O/id_share.json --against $O/id_main_omp112.json 2>&1 | grep -E "serial|threads|Error|Traceback") \
  > $O/share_identity.log
log "identity done"

twocard() {
  local name=$1 tree=$2
  quiet_host
  note $name $tree "two-card ci video c1 c16"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_disagg_ci --port 28400 --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
twocard tc_s1 $SHARE
twocard tc_s2 $SHARE

mixed() {
  local name=$1 tree=$2 card=$3
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  note $name $tree "seedtts c16 beside text video card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$CONTEXT_32K" > $O/$name.driver.log 2>&1 &
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
both "mixed mx1_l0 $LANE 0" "mixed mx1_s1 $SHARE 1"
quiet_host
both "mixed mx2_s0 $SHARE 0" "mixed mx2_l1 $LANE 1"
log "q1115_share done"
