#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_m: does capacity close the speech underruns #3452 exposes on a saturated
# card. The traced pair (q1128d_l) put the cost in talker rows excluded from decode steps while they wait for thinker text
# (26 % of a stream's steps on dev2 against 12 % on main) and in long gaps where the thinker's 230 ms video prefill steps
# share the card with the talker. STACK dev/qwen3-omni-stack-1128 f3e19a057 (main, part 2: thinker, talker and code2wav
# in one process and context, the full prefill graph, and #3452 91c2e2010) against MAIN, H20 emulation, speech c16
# closed loop beside 8 looping videos, same-time pairs swapped.
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main S=/workspace/stack
STACK_COMMIT=f3e19a057
STACK_TREE=8e9a6fa3a230
H20_FREE_MIB=97871
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
closed() {
  local name=$1 tree=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  mkdir -p $O/$name
  python3 $K/ballast.py --card $card --free-mib $H20_FREE_MIB --ready-file $O/$name/ballast.txt > $O/$name/ballast.log 2>&1 &
  local ballast=$!
  until [ -s $O/$name/ballast.txt ] || ! kill -0 $ballast 2>/dev/null; do sleep 2; done
  note $name $tree "H20 colocated bytes on card $card, speech c16 closed loop beside 8 looping videos"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated_h20_on_h200 --card $card --port $port \
    --out $O/$name --loads 16 --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port $cpus &
  local looper=$!
  wait $driver
  local rc=$?
  wait $looper
  kill $ballast 2>/dev/null
  wait $ballast 2>/dev/null
  log "boot $name rc $rc oom $(grep -c -i -E 'out of memory|OutOfMemoryError' $O/$name/serve.log)"
}
until grep -q "q1128d_m done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_n start"
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-stack-1128
git worktree add -q --detach $S $STACK_COMMIT
log "tree stack $(git -C $S rev-parse --short HEAD) $(git -C $S rev-parse --short=12 HEAD:sglang_omni) want $STACK_TREE"
if [ "$(git -C $S rev-parse --short=12 HEAD:sglang_omni)" != "$STACK_TREE" ]; then log "stack tree mismatch, stop"; exit 1; else :; fi
both "closed hs1_m $M 0 1800" "closed hs1_s $S 1 1800"
both "closed hs2_s $S 0 2000" "closed hs2_m $M 1 2000"
python3 $K/playback_margin.py $O/hs1_m $O/hs2_m $O/hs1_s $O/hs2_s >> $O/queue.log 2>&1
python3 $K/body_table.py $O c16 "the stack with #3452 against main, H20 emulation, speech c16 closed loop beside 8 looping videos" \
  main=hs1_m,hs2_m fix=hs1_s,hs2_s >> $O/queue.log 2>&1
log "h20 stack read"
log "q1128d_n done"
