#!/usr/bin/env bash
# 11-22, node-0001, card 0 after q1122_a's seeded boots. Part 2 now yields the turn between a step's selection, build
# and forward (184e443a2): in one context at c16 the talker waited 174 ms a second behind thinker holds of 20 to 26 ms
# on MIXED steps. Combined with it: OCVG3 84eed4847 (tree 62a2ba69a8e1) against part 2's base BVG 6aff09f46 (tree
# 32cb9d81e911), card 0 alone, B A A B:
#   Y1  speech alone c1 and c16 (part 2's c16 first audio p99 was +13.0 %, inter chunk p99 +3.9 %)
#   Y2  speech 8/s seed 3451 beside video 1/s seed 7 (first audio p99 was +2.9 %), every video boot gated
#   X   the index copy fix alone (IXC 4ca23252e, tree 03e7268c8857) against main (NMAIN e0482a08c), speech alone c16 and
#       c64, for its PR
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain BVG=/workspace/bvg OCVG3=/workspace/ocvg3 IXC=/workspace/ixc
OCVG3_SHA=84eed4847 IXC_SHA=4ca23252e
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
  echo $((28000 + 110 + 200 * (BOOTS % 20)))
}
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
}
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
  if [ -n "$p50" ] && [ "$p50" -le 649 ]; then
    log "$line, within 649"
  else
    log "$line, over 649: void, rerun as ${name}r, $(node_free)"
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
until grep -q "^W boots done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 0 " > /dev/null; do sleep 20; done
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-one-context-visual-graph-read dev/thinker-index-copies-without-stream-wait
git worktree add -q --detach $OCVG3 FETCH_HEAD 2>/dev/null || true
git -C $OCVG3 fetch -q origin dev/qwen3-omni-one-context-visual-graph-read && git -C $OCVG3 checkout -q --detach FETCH_HEAD
git -C /workspace/sglang-omni fetch -q origin dev/thinker-index-copies-without-stream-wait
git worktree add -q --detach $IXC FETCH_HEAD 2>/dev/null || true
git -C $IXC fetch -q origin dev/thinker-index-copies-without-stream-wait && git -C $IXC checkout -q --detach FETCH_HEAD
for pair in "$OCVG3 $OCVG3_SHA" "$IXC $IXC_SHA"; do
  set -- $pair
  if [ "$(git -C $1 rev-parse --short=9 HEAD)" != "$2" ] || ! git -C $1 diff --quiet HEAD -- sglang_omni; then
    log "q1122_c stopped: $1 is not a clean $2"
    exit 1
  else :; fi
done
log "q1122_c start, $(node_free)"
alone yb1 $OCVG3 1,16
alone ya2 $BVG 1,16
alone ya3 $BVG 1,16
alone yb4 $OCVG3 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "part 2 with step yields (84eed4847) against its base (6aff09f46), card 0 B A A B, speech alone $cell" \
    two=ya2,ya3 one=yb1,yb4 >> $O/queue.log 2>&1
done
python3 $K/first_audio_states.py --cell c16_events --main $O/ya2,$O/ya3 --branch $O/yb1,$O/yb4 > $O/y1_states_c16.txt 2>&1
grep -E "server first audio|frames until|thinker prefill|thinker build" $O/y1_states_c16.txt >> $O/queue.log
log "Y1 read"
mixed ym1_b $OCVG3
b1=$LAST
mixed ym2_a $BVG
a2=$LAST
mixed ym3_a $BVG
a3=$LAST
mixed ym4_b $OCVG3
b4=$LAST
python3 $K/body_table.py $O c64 "part 2 with step yields against its base, card 0 B A A B, speech 8/s and video 1/s" \
  two=$a2,$a3 one=$b1,$b4 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O two=$a2,$a3 one=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "Y2 read"
alone xb1 $IXC 16,64
alone xa2 $NMAIN 16,64
alone xa3 $NMAIN 16,64
alone xb4 $IXC 16,64
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "index copy fix alone (4ca23252e) against main (e0482a08c), card 0 B A A B, speech alone $cell" \
    main=xa2,xa3 fix=xb1,xb4 >> $O/queue.log 2>&1
done
log "X read"
log "q1122_c done"
