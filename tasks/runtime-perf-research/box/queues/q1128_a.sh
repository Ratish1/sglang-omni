#!/usr/bin/env bash
# 11-28, node-0001, both H200s, same-time pairs (card 0 on CPUs 0-55,112-167, card 1 on 56-111,168-223), swapped.
# #3452 alone moves video time from preprocessing into the thinker's decode (11-27 traces: video thinker decode +29.5 %
# cached, +10.3 % uncached, because speech no longer waits behind video), so its other worse rows are owned by the
# thinker and the shared card. This reads the capacity stack with it:
#   M  upstream main 0f70d9c7c
#   S0 main + part 2 + the full prefill graph (5ee1c55f2), dev/qwen3-omni-stack-1128-no3452 f3ac01ef4
#   S  S0 + #3452 (91c2e2010), dev/qwen3-omni-stack-1128 f3e19a057
# Cells: the affected suites on S; speech 8/s beside video 1/s colocated, cached (M S, S M, S0 S, S S0) and uncached
# (thinker radix cache off: M S, S M); speech alone c16 and c64 (M S, S M); the Video-MME gates on fresh servers
# (text c16 once, talker once, 10 rounds M S swapped).
set -u
O=/data/v1128
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main S0=/workspace/stack0 S=/workspace/stack
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
UNCACHED="--thinker.engine.disable_radix_cache true"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
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
until grep -q "setup done" $O/queue.log 2>/dev/null; do sleep 20; done
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-stack-1128-no3452 dev/qwen3-omni-stack-1128
git worktree add -q --detach $S0 origin/dev/qwen3-omni-stack-1128-no3452
git worktree add -q --detach $S origin/dev/qwen3-omni-stack-1128
for tree in $S0 $S; do log "tree $tree $(git -C $tree rev-parse --short HEAD) $(git -C $tree rev-parse --short=12 HEAD:sglang_omni)"; done
log "q1128_a start"

cd $S
PYTHONPATH=$S taskset -c $(cpus_of 0) python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_omni \
  tests/unit_test/preprocessing tests/unit_test/pipeline tests/unit_test/scheduling tests/unit_test/model_runner \
  tests/unit_test/relay tests/unit_test/minicpm_o tests/unit_test/ming_omni > $O/units_stack.log 2>&1
log "units stack rc $? $(grep -E '[0-9]+ (passed|failed)' $O/units_stack.log | tail -1)"

export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
video_loop() {
  local name=$1 tree=$2 driver=$3 port=$4 cpus=$5
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 --request-rate 1 \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
}
mixed() {
  local name=$1 tree=$2 card=$3 offset=$4 extra=$5
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "colocated, speech 8/s beside video 1/s, card $card, $extra"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K $extra" --omni-chunk-audio "--bench-extra=--request-rate=8" \
    --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port $cpus &
  local looper=$!
  wait $driver
  log "boot $name rc $? $(grep -m1 -o -E 'Traceback|Error[^ ]*' $O/$name/serve.log 2>/dev/null)"
  wait $looper
}
speech() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "colocated, speech alone c16 and c64, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16,64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -m1 -o -E 'Traceback|Error[^ ]*' $O/$name/serve.log 2>/dev/null)"
}
fresh() {
  local name=$1 tree=$2 card=$3 offset=$4
  local port=$(port_of $card $offset)
  note $name $tree "CI speech pipeline, fresh server, text Video-MME c16 once, talker once, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card $card --port $port \
    --out $O/$name --bench videomme_text --loads 16 --passes clean \
    --after "python3 $K/videomme_talker_ci.py --port $port --out $O/$name/talker --passes 1" > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -o '\"accuracy\": [0-9.]*' $O/$name/talker/summary.jsonl 2>/dev/null | head -1)"
}

quiet_host
both "mixed c1_m $M 0 100 ''" "mixed c1_s $S 1 100 ''"
both "mixed c2_s $S 0 300 ''" "mixed c2_m $M 1 300 ''"
both "mixed c3_s0 $S0 0 500 ''" "mixed c3_s $S 1 500 ''"
both "mixed c4_s $S 0 700 ''" "mixed c4_s0 $S0 1 700 ''"
python3 $K/body_table.py $O c64 "stack with #3452 against main, cached, 8/s beside video 1/s" main=c1_m,c2_m \
  stack=c1_s,c2_s >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "stack with #3452 against the stack without it, cached" stack0=c3_s0,c4_s0 \
  stack=c3_s,c4_s >> $O/queue.log 2>&1
log "cached read"
quiet_host
both "mixed u1_m $M 0 900 '$UNCACHED'" "mixed u1_s $S 1 900 '$UNCACHED'"
both "mixed u2_s $S 0 1100 '$UNCACHED'" "mixed u2_m $M 1 1100 '$UNCACHED'"
python3 $K/body_table.py $O c64 "stack with #3452 against main, uncached, 8/s beside video 1/s" main=u1_m,u2_m \
  stack=u1_s,u2_s >> $O/queue.log 2>&1
log "uncached read"
quiet_host
both "speech a1_m $M 0 1300" "speech a1_s $S 1 1300"
both "speech a2_s $S 0 1500" "speech a2_m $M 1 1500"
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "stack with #3452 against main, speech alone, $cell" main=a1_m,a2_m \
    stack=a1_s,a2_s >> $O/queue.log 2>&1
done
log "speech alone read"
for round in 1 2 3 4 5 6 7 8 9 10; do
  if [ $((round % 2)) = 1 ]; then
    both "fresh g${round}_m $M 0 $((1700 + 200 * round))" "fresh g${round}_s $S 1 $((1700 + 200 * round))"
  else
    both "fresh g${round}_s $S 0 $((1700 + 200 * round))" "fresh g${round}_m $M 1 $((1700 + 200 * round))"
  fi
done
mains=$(ls -d $O/g*_m | xargs -n1 basename | sed 's|$|/c16_clean|' | paste -sd,)
stacks=$(ls -d $O/g*_s | xargs -n1 basename | sed 's|$|/c16_clean|' | paste -sd,)
python3 $K/videomme_answer_diff.py $O main=$mains stack=$stacks > $O/gate_text.txt 2>&1
mains=$(ls -d $O/g*_m | xargs -n1 basename | sed 's|$|/talker|' | paste -sd,)
stacks=$(ls -d $O/g*_s | xargs -n1 basename | sed 's|$|/talker|' | paste -sd,)
python3 $K/videomme_answer_diff.py $O main=$mains stack=$stacks > $O/gate_talker.txt 2>&1
log "gates read"
log "q1128_a done"
