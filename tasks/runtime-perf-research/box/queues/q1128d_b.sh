#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_a: the checks of tasks/pr3452_91c2e201/CORRECTNESS.md that decide items
# a to e, same-time pairs on the two cards, swapped, FIX 91c2e2010 against MAIN 0f70d9c7c.
#   1. Check 1, the batch-invariant mode as a diagnostic (never serving or CI): the thinker with SGLang's deterministic
#      inference, everything else default (mixed chunk, radix cache, graphs on), CI Video-MME text c16 x3 and the talker
#      test x3. Every response identical across passes and arms means all default-serving movement is kernel order.
#   2. Check 7, c1 identity with the thinker's radix cache off: text Video-MME c1 x2 on fresh servers.
#   3. Check 6, the removed opt-in, main only: MMSU (2,000 questions) on the fused text pipeline c96, main default against
#      main with --preprocessing.factory.max_concurrency 4.
#   4. Check 8, speech c16 closed loop beside 8 looping videos, colocated, with the image encoder's GPU memory sampled
#      every 100 ms (check 3's bytes at the batches the PR forms).
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main F=/workspace/fix
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
DETERMINISTIC="--thinker.engine.enable_deterministic_inference true --thinker.engine.attention_backend fa3"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
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
until grep -q "q1128d_a done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_b start"

deterministic() {
  local name=$1 tree=$2 card=$3 offset=$4
  local port=$(port_of $card $offset)
  note $name $tree "CI speech pipeline, thinker deterministic, text c16 x3, talker x3, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card --port $port \
    --out $O/$name --serve-extra "$DETERMINISTIC" --bench videomme_text --loads 16 --passes clean --bench-repeats 3 \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 3" > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -m1 -o -E 'deterministic[^,]*' $O/$name/serve.log | head -1)"
}
both "deterministic dt1_m $M 0 1100" "deterministic dt1_f $F 1 1100"
both "deterministic dt2_f $F 0 1300" "deterministic dt2_m $M 1 1300"
for part in c16_clean talker; do
  python3 $K/videomme_answer_diff.py $O main=dt1_m/$part,dt2_m/$part fix=dt1_f/$part,dt2_f/$part > $O/det_$part.txt 2>&1
done
log "deterministic read: text $(grep -E '^answers' $O/det_c16_clean.txt), talker $(grep -E '^answers' $O/det_talker.txt)"

lone_uncached() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "CI speech pipeline, thinker radix cache off, text c1 x2, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card \
    --port $(port_of $card $offset) --out $O/$name --serve-extra "--thinker.engine.disable_radix_cache true" \
    --bench videomme_text --loads 1 --passes clean --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both "lone_uncached lu1_m $M 0 1500" "lone_uncached lu1_f $F 1 1500"
both "lone_uncached lu2_f $F 0 1700" "lone_uncached lu2_m $M 1 1700"
python3 $K/videomme_answer_diff.py $O main=lu1_m/c1_clean,lu2_m/c1_clean fix=lu1_f/c1_clean,lu2_f/c1_clean > $O/lone_uncached.txt 2>&1
log "c1 uncached read: $(grep -E '^answers|stable in' $O/lone_uncached.txt | tr '\n' ' ')"

mmsu() {
  local name=$1 card=$2 offset=$3 extra=$4
  note $name $M "fused text pipeline, MMSU c96 and c16, $extra, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $M --model omni_text_fused --card $card \
    --port $(port_of $card $offset) --out $O/$name --serve-extra "$extra" --bench mmsu --loads 96,16 --passes clean \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -m1 'Throughput:' $O/$name/c96_clean/bench.log 2>/dev/null)"
}
both "mmsu ms1_d 0 1900 ''" "mmsu ms1_o 1 1900 '--preprocessing.factory.max_concurrency 4'"
both "mmsu ms2_o 0 2100 '--preprocessing.factory.max_concurrency 4'" "mmsu ms2_d 1 2100 ''"
log "mmsu opt-in read"

export OMNI_ARRIVAL_SEED=3451
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
gpu_sampler() {
  local name=$1 card=$2 driver=$3
  while kill -0 $driver 2>/dev/null; do
    nvidia-smi -i $card --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | sed "s/^/$(date +%s.%N | cut -c1-14),/" >> $O/$name/gpu_samples.csv
    sleep 0.1
  done
}
closed() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  mkdir -p $O/$name
  note $name $tree "colocated, speech c16 closed loop beside 8 looping videos, card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port $cpus &
  local looper=$!
  gpu_sampler $name $card $driver &
  local sampler=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper $sampler
}
both "closed cl1_m $M 0 2300" "closed cl1_f $F 1 2300"
both "closed cl2_f $F 0 2500" "closed cl2_m $M 1 2500"
python3 $K/body_table.py $O c16 "the PR against main, speech c16 closed loop beside 8 looping videos, colocated" \
  main=cl1_m,cl2_m fix=cl1_f,cl2_f >> $O/queue.log 2>&1
log "closed loop read"
log "q1128d_b done"
