#!/usr/bin/env bash
# 11-20, node-0000, both H200s: the #3452 rework (vlane 9ff361195) against main (4be177591) as same-time pairs. One
# instance per card, each pinned to its own socket (card 0 CPUs 0-55,112-167, card 1 CPUs 56-111,168-223; sysfs puts
# both cards on NUMA 0, so card 1's server keeps its socket), arm orders opposite, so every round runs main on one card
# and the rework on the other, and the next round swaps them. Each round starts when both cards finished the last.
# Cells: the CI speech pipeline on one card, text Video-MME c1 and c16 and the CI talker test (oc); SeedTTS c16 beside
# 8 looping Video-MME videos on that server (sv) and on the colocated config (cv).
# usage: q1120_gpu.sh <card> <cpus> "<arm order>"
set -u
CARD=$1 CPUS=$2 ORDER=$3
O=/data/v1120/gpu
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/main LANE=/workspace/vlane
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
BASE_PORT=$((28000 + 1000 * CARD))
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
mkdir -p $O/sync
log() { echo "$* card$CARD $(date -u +%T)" >> /data/v1120/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
tree_of() { case $1 in m) echo $MAIN ;; l) echo $LANE ;; *) exit 2 ;; esac; }
sync_round() {
  touch $O/sync/$1.card$CARD
  until [ -e $O/sync/$1.card$((1 - CARD)) ]; do sleep 10; done
}

ci_text_and_talker() {
  local name=$1 tree=$2 port=$((BASE_PORT + 400))
  note $name $tree "omni_ci_onecard text video c1 c16, talker c16"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $CARD --port $port --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean,events --bench-repeats 2 \
    --after "python3 $K/stage_cpus.py \$(cat $O/$name/server.pgid) > $O/$name/stage_cpus.txt; python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 3" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}

mixed() {
  local name=$1 tree=$2 model=$3 extra=$4 port=$5
  note $name $tree "$model seedtts c16 beside 8 videos"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model $model --card $CARD --port $port --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$extra" > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
    cd $tree
    pass=0
    while kill -0 $driver 2>/dev/null; do
      PYTHONPATH=$tree taskset -c $CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
        --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
        --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 \
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

until grep -q "q1120_cpu done" /data/v1120/queue.log; do sleep 30; done
log "q1120_gpu start order $ORDER cpus $CPUS"
round=0
for cell in oc sv cv; do
  for arm in $ORDER; do
    round=$((round + 1))
    sync_round r$round
    index=$(ls -d $O/${cell}_${arm}*_c$CARD 2>/dev/null | wc -l)
    name=${cell}_${arm}$((index + 1))_c$CARD
    case $cell in
      oc) ci_text_and_talker $name $(tree_of $arm) ;;
      sv) mixed $name $(tree_of $arm) omni_ci_onecard "" $((BASE_PORT + 500)) ;;
      cv) mixed $name $(tree_of $arm) omni_colocated "$CONTEXT_32K" $((BASE_PORT + 600)) ;;
    esac
  done
done
log "q1120_gpu done"
