#!/usr/bin/env bash
# 11-22, node-0001. Part 2 with both turn rules (9ee358305): the thinker ranks first while a speaking request it holds
# awaits its first emit, and the talker ranks last while one of its rows waits on thinker text. Combined with them:
# OCVG5 46b5e1206 (tree 61627b4ea25c) against part 2's base BVG 6aff09f46, B A A B.
#   G1 (card 1)  speech alone c1 and c16, then a traced c16 boot
#   G2 (card 0, after q1122_c)  speech 8/s beside video 1/s, every video boot gated; then speech alone c64
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
BVG=/workspace/bvg OCVG5=/workspace/ocvg5
OCVG5_SHA=46b5e1206
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
TRACE_PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_TURNS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
cpus_of() { [ "$1" = 0 ] && echo 0-55,112-167 || echo 56-111,168-223; }
alone() {
  local name=$1 tree=$2 loads=$3 card=$4 port=$5
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts loads $loads alone card $card port $port"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port \
    --out $O/$name --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video_loop() {
  local name=$1 tree=$2 port=$3 driver=$4
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $(cpus_of 0) python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
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
  local name=$1 tree=$2 port=$3
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s card 0 port $port"
  taskset -c $(cpus_of 0) python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
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
MIXED_LAST=""
mixed() {
  local name=$1 tree=$2 port=$3
  mixed_once $name $tree $port
  MIXED_LAST=$name
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  if [ -n "$p50" ] && [ "$p50" -le 649 ]; then
    log "$line, within 649"
  else
    log "$line, over 649: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree $port
    MIXED_LAST=${name}r
    log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O ${name}r)"
  fi
}
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-one-context-visual-graph-read
git worktree add -q --detach $OCVG5 FETCH_HEAD
if [ "$(git -C $OCVG5 rev-parse --short=9 HEAD)" != "$OCVG5_SHA" ] || ! git -C $OCVG5 diff --quiet HEAD -- sglang_omni; then
  log "q1122_g stopped: ocvg5 is not a clean $OCVG5_SHA"
  exit 1
else :; fi
log "q1122_g start, $(node_free)"
(
  until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
  alone gb1 $OCVG5 1,16 1 31210
  alone ga2 $BVG 1,16 1 31210
  alone ga3 $BVG 1,16 1 31210
  alone gb4 $OCVG5 1,16 1 31210
  for cell in c1 c16; do
    python3 $K/body_table.py $O $cell "part 2 with both turn rules (46b5e1206) against its base, card 1 B A A B, speech alone $cell" \
      two=ga2,ga3 one=gb1,gb4 >> $O/queue.log 2>&1
  done
  python3 $K/first_audio_states.py --cell c16_events --main $O/ga2,$O/ga3 --branch $O/gb1,$O/gb4 > $O/g_states_c16.txt 2>&1
  grep -E "server first audio|frames until|thinker prefill|thinker build|hop to the thinker" $O/g_states_c16.txt >> $O/queue.log
  log "G1 read"
  log "boot tb4 $(git -C $OCVG5 rev-parse --short HEAD) traced c16 card 1 start"
  taskset -c $(cpus_of 1) python3 $K/baseline.py --tree $OCVG5 --model omni_colocated --card 1 --port 31250 --out $O/tb4 \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events --probe --serve-env "$TRACE_PROBES" \
    > $O/tb4.driver.log 2>&1
  log "boot tb4 rc $?"
  python3 $K/first_window_feed.py $O/tb4 --cell c16_events > $O/tb4.feed.txt 2>&1
  python3 $K/turn_waits.py $O/tb4 --cell c16_events > $O/tb4.turns_c16.txt 2>&1
  grep -h "slowest\|all (median)" $O/tb4.feed.txt >> $O/queue.log
  grep -h "waiter priority" $O/tb4.turns_c16.txt >> $O/queue.log
  log "G1T read"
) &
card1=$!
until grep -q "^q1122_c done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 0 " > /dev/null; do sleep 20; done
mixed gm1_b $OCVG5 28410
b1=$MIXED_LAST
mixed gm2_a $BVG 28410
a2=$MIXED_LAST
mixed gm3_a $BVG 28410
a3=$MIXED_LAST
mixed gm4_b $OCVG5 28410
b4=$MIXED_LAST
python3 $K/body_table.py $O c64 "part 2 with both turn rules against its base, card 0 B A A B, speech 8/s and video 1/s" \
  two=$a2,$a3 one=$b1,$b4 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O two=$a2,$a3 one=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "G2 mixed read"
alone gc1 $OCVG5 64 0 28450
alone gc2 $BVG 64 0 28450
alone gc3 $BVG 64 0 28450
alone gc4 $OCVG5 64 0 28450
python3 $K/body_table.py $O c64 "part 2 with both turn rules against its base, card 0 B A A B, speech alone c64" \
  two=gc2,gc3 one=gc1,gc4 >> $O/queue.log 2>&1
log "G2 c64 read"
wait $card1
log "q1122_g done"
