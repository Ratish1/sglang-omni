#!/usr/bin/env bash
# 11-21, node-0001, card 0 after q1121_final2's F2. The answer to the original question in its final form: #3452's stack
# (preprocessing video lane and talker slot state) with both changes, part 1, part 2 and the index copy fix (SOCVG
# 26224b691, tree 26547072a361) against main, card 0 alone, B A A B, speech 8/s seed 3451 beside video 1/s seed 7 (cap 8), every video
# boot gated (stack-based trees 445 ms, main 580); then speech alone c16, B A A B.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain SOCVG=/workspace/socvg
SOCVG_SHA=26224b691
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=0-55,112-167
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
BOOTS=0
LAST=""
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
next_port() {
  BOOTS=$((BOOTS + 1))
  echo $((28000 + 130 + 200 * (BOOTS % 20)))
}
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
}
expected_ms() { case $1 in $SOCVG) echo 445 ;; *) echo 580 ;; esac; }
video_loop() {
  local name=$1 tree=$2 port=$3 driver=$4
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
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
mixed_once() {
  local name=$1 tree=$2
  local port=$(next_port)
  quiet_host
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s card 0 port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $port $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
mixed() {
  local name=$1 tree=$2
  mixed_once $name $tree
  LAST=$name
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  local limit=$(($(expected_ms $tree) * 112 / 100))
  if [ -n "$p50" ] && [ "$p50" -le "$limit" ]; then
    log "$line, within $limit"
  else
    log "$line, over $limit: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree
    LAST=${name}r
    log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O ${name}r)"
  fi
}
alone() {
  local name=$1 tree=$2 loads=$3
  local port=$(next_port)
  quiet_host
  note $name $tree "seedtts loads $loads alone card 0 port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "^F2 read" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 0 " > /dev/null; do sleep 20; done
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-stack-one-context-visual-graph
git worktree add -q --detach $SOCVG FETCH_HEAD
log "tree socvg $(git -C $SOCVG rev-parse --short HEAD) $(git -C $SOCVG rev-parse --short=12 HEAD:sglang_omni)"
if [ "$(git -C $SOCVG rev-parse --short=9 HEAD)" != "$SOCVG_SHA" ]; then
  log "q1121_stackoc stopped: socvg is not $SOCVG_SHA"
  exit 1
else :; fi
log "q1121_stackoc start, $(node_free)"
mixed sm1_b $SOCVG
b1=$LAST
mixed sm2_a $NMAIN
a2=$LAST
mixed sm3_a $NMAIN
a3=$LAST
mixed sm4_b $SOCVG
b4=$LAST
python3 $K/body_table.py $O c64 "#3452 stack with both, part 1 and part 2 against main, card 0 B A A B, speech 8/s and video 1/s" \
  main=$a2,$a3 stack_full=$b1,$b4 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O main=$a2,$a3 stack_full=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "S1 read"
alone ss1_b $SOCVG 16
alone ss2_a $NMAIN 16
alone ss3_a $NMAIN 16
alone ss4_b $SOCVG 16
python3 $K/body_table.py $O c16 "#3452 stack full against main, card 0 B A A B, speech alone c16" \
  main=ss2_a,ss3_a stack_full=ss1_b,ss4_b >> $O/queue.log 2>&1
log "S2 read"
log "q1121_stackoc done"
