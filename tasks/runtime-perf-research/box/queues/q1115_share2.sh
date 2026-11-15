#!/usr/bin/env bash
# 11-15, node-0000, both H200s: the video CPU share only where the config splits the team (SHARE2 cea7dde64; a configured
# team, as colocated sets, stays each video's team, which the colocated speech beside video pairs showed SHARE 599c63d24
# costing first audio +7 % against the previous head). Units; colocated SeedTTS c16 beside 4 videos, LANE (ecfebb1f3)
# against SHARE2 as same-time pairs swapped; one CI two-card Video-MME boot to see the config hand the factory its
# CPUs; SeedTTS c16 beside 4 videos on the CI two-card speech server (both cards, whole host), boots l b b l; the stage's
# memory and threads at 8 video threads, LANE against SHARE2.
set -u
O=/data/v1115
K=/workspace/kit2/tasks/runtime-perf-research/box
SHARE2=/workspace/vshare2 LANE=/workspace/vlane
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
CPUS1=56-111,168-223
ALL=0-223
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
log "q1115_share2 start"

(cd $SHARE2 && python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni tests/unit_test/preprocessing \
  tests/unit_test/utils tests/unit_test/minicpm_o tests/unit_test/ming_omni 2>&1 | tail -15) > $O/share2_units.log
log "units $(grep -E 'passed|failed' $O/share2_units.log | tail -1)"

mixed() {
  local name=$1 tree=$2 card=$3 model=$4
  local cpus=$CPUS0 port=28400 extra="$CONTEXT_32K"
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  if [ $model = omni_disagg_ci ]; then cpus=$ALL; extra=""; else :; fi
  note $name $tree "$model seedtts c16 beside text video card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$extra" > $O/$name.driver.log 2>&1 &
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
both "mixed mb1_l0 $LANE 0 omni_colocated" "mixed mb1_b1 $SHARE2 1 omni_colocated"
quiet_host
both "mixed mb2_b0 $SHARE2 0 omni_colocated" "mixed mb2_l1 $LANE 1 omni_colocated"
log "colocated pairs done"

quiet_host
note tc_b1 $SHARE2 "two-card ci video c1 c16"
taskset -c $CPUS0 python3 $K/baseline.py --tree $SHARE2 --model omni_disagg_ci --port 28400 --out $O/tc_b1 \
  --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 > $O/tc_b1.driver.log 2>&1
log "boot tc_b1 rc $?"

for arm in md1_l:$LANE md1_b:$SHARE2 md2_b:$SHARE2 md2_l:$LANE; do
  quiet_host
  mixed ${arm%%:*} ${arm#*:} 0 omni_disagg_ci
done
log "two-card mixed done"

for arm in lane:$LANE share2:$SHARE2; do
  (cd ${arm#*:} && PYTHONPATH=${arm#*:} taskset -c $CPUS0 python3 $K/preprocess_memory.py --threads 8 --concurrency 16 \
    --video-cpus $([ ${arm%%:*} = share2 ] && echo 112 || echo 0) --out $O/memory_${arm%%:*}.json 2>&1 \
    | grep -E "threads|Error|Traceback") >> $O/share2_memory.log
done
log "q1115_share2 done"
