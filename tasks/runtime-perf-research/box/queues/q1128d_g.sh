#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_f: DEV2 (dev/qwen3-omni-preprocessing-request-threads 4cac082f9, source
# tree d5b2daf9ff43: the PR head, requests without video on max_concurrency threads again, and each request with video
# encoded alone in the image encoder) against FIX 91c2e2010 and MAIN 0f70d9c7c.
#   1. The new image encoder test on the PR head (must fail), then the scheduler and encoder tests on the dev2 tree.
#   2. The H20 colocated config's bytes under a ballast (q1128d_e's emulation) with the heaviest load seen: speech c16
#      closed loop beside 8 looping videos. Pairs FIX against DEV2 twice (swapped), then MAIN against DEV2.
#   3. The same emulation with CI Video-MME text c16 x2 and the talker test, MAIN against DEV2 (q1128d_e's cell).
#   4. Speed on one H200 without a ballast: CI Video-MME text c16 x2 from a fresh boot, FIX against DEV2, 2 pairs.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main F=/workspace/fix V=/workspace/dev2 P=/workspace/prefix
DEV2_COMMIT=4cac082f9
DEV2_TREE=d5b2daf9ff43
H20_FREE_MIB=97871
TESTS=tests/unit_test/qwen3_omni/test_pipeline.py
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
gpu_sampler() {
  local name=$1 card=$2 driver=$3
  while kill -0 $driver 2>/dev/null; do
    nvidia-smi -i $card --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | sed "s/^/$(date +%s.%N | cut -c1-14),/" >> $O/$name/gpu_samples.csv
    sleep 0.1
  done
}
video_loop() {
  local name=$1 tree=$2 driver=$3 port=$4 cpus=$5
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
}
start_ballast() {
  local name=$1 card=$2
  mkdir -p $O/$name
  python3 $K/ballast.py --card $card --free-mib $H20_FREE_MIB --ready-file $O/$name/ballast.txt > $O/$name/ballast.log 2>&1 &
  BALLAST=$!
  until [ -s $O/$name/ballast.txt ] || ! kill -0 $BALLAST 2>/dev/null; do sleep 2; done
}
oom_line() {
  local name=$1
  echo "oom $(grep -c -i -E 'out of memory|OutOfMemoryError' $O/$name/serve.log) video failed $(grep -h -o -E '"failed": [0-9]+' $O/$name/video_pass*/*.json $O/$name/c16_clean/bench/rep*/videomme_results.json $O/$name/talker/summary.jsonl 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
}
until grep -q "q1128d_f done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_g start"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-preprocessing-request-threads
git worktree add -q --detach $V $DEV2_COMMIT
log "tree dev2 $(git -C $V rev-parse --short HEAD) $(git -C $V rev-parse --short=12 HEAD:sglang_omni) want $DEV2_TREE"
if [ "$(git -C $V rev-parse --short=12 HEAD:sglang_omni)" != "$DEV2_TREE" ]; then log "dev2 tree mismatch, stop"; exit 1; else :; fi

cp $V/$TESTS $P/$TESTS
cd $P && python3 -m pytest -q -p no:cacheprovider $TESTS -k "test_image_encoder_batches_images_together_and_each_video_alone" > $O/unit_prefix2.log 2>&1
log "unit prefix2 (must fail) rc $? $(tail -1 $O/unit_prefix2.log)"
cd $V && python3 -m pytest -q -p no:cacheprovider $TESTS -k "preprocessing or video_lane or read_request_inputs or image_encoder" > $O/unit_dev2.log 2>&1
log "unit dev2 rc $? $(tail -1 $O/unit_dev2.log)"

h20_closed() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  start_ballast $name $card
  note $name $tree "H20 colocated bytes on card $card, $(cat $O/$name/ballast.txt 2>/dev/null), speech c16 closed loop beside 8 looping videos"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated_h20_on_h200 --card $card --port $port \
    --out $O/$name --loads 16 --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  gpu_sampler $name $card $driver &
  local sampler=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port $cpus &
  local looper=$!
  wait $driver
  local rc=$?
  wait $looper $sampler
  kill $BALLAST 2>/dev/null
  wait $BALLAST 2>/dev/null
  log "boot $name rc $rc $(oom_line $name)"
}
both "h20_closed hc1_f $F 0 4900" "h20_closed hc1_v $V 1 4900"
both "h20_closed hc2_v $V 0 5100" "h20_closed hc2_f $F 1 5100"
both "h20_closed hc3_m $M 0 5300" "h20_closed hc3_v $V 1 5300"
python3 $K/playback_margin.py $O/hc1_f $O/hc2_f $O/hc1_v $O/hc2_v $O/hc3_v $O/hc3_m >> $O/queue.log 2>&1
log "h20 closed read"

h20_text() {
  local name=$1 tree=$2 card=$3 offset=$4
  local port=$(port_of $card $offset)
  start_ballast $name $card
  note $name $tree "H20 colocated bytes on card $card, $(cat $O/$name/ballast.txt 2>/dev/null), text c16 x2, talker x1"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated_h20_on_h200 --card $card \
    --port $port --out $O/$name --bench videomme_text --loads 16 --passes clean --bench-repeats 2 \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 1" > $O/$name.driver.log 2>&1 &
  local driver=$!
  gpu_sampler $name $card $driver &
  local sampler=$!
  wait $driver
  local rc=$?
  wait $sampler
  kill $BALLAST 2>/dev/null
  wait $BALLAST 2>/dev/null
  log "boot $name rc $rc $(oom_line $name)"
}
both "h20_text h3_m $M 0 5500" "h20_text h3_v $V 1 5500"
log "h20 text read"

speed() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "CI one card, text c16 x2 from a fresh boot, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card \
    --port $(port_of $card $offset) --out $O/$name --bench videomme_text --loads 16 --passes clean --bench-repeats 2 \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both "speed sp1_f $F 0 5700" "speed sp1_v $V 1 5700"
both "speed sp2_v $V 0 5900" "speed sp2_f $F 1 5900"
log "speed read"
log "q1128d_g done"
