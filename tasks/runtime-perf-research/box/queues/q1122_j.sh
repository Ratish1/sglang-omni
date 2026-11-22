#!/usr/bin/env bash
# 11-22, node-0001, card 0. Part 2's final turn order (240338e2c): the talker first; a talker row waiting on thinker
# text makes the talker defer; a backlogged talker (requests still waiting after its selection) is expedited and never
# defers. The thinker's own first emit expedite is dropped (c16 first audio p99 +18.8 % with it, both boots).
# Combined: OCVG7 085064564 (tree a64419ba43de) against part 2's base BVG 6aff09f46, B A A B.
#   J1 speech alone c1 and c16   J2 speech 8/s beside video 1/s, gated   J3 speech alone c64
#   J4 seeded c1 wav identity: OCVG7 against BVG, the index copy fix (IXC 4ca23252e) against main (NMAIN e0482a08c)
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain BVG=/workspace/bvg OCVG7=/workspace/ocvg7 IXC=/workspace/ixc
OCVG7_SHA=085064564
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS0=0-55,112-167
PORT=28570
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
alone() {
  local name=$1 tree=$2 loads=$3
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts loads $loads alone card 0 port $PORT"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $PORT --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
seeded() {
  local name=$1 tree=$2
  note $name $tree "seeded c1 card 0 port $PORT"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $PORT --out $O/$name \
    --loads 1 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes clean "--bench-extra=--seed 3451" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video_loop() {
  local name=$1 tree=$2 driver=$3
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS0 python3 $K/videomme_speed.py --host 127.0.0.1 --port $PORT \
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
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s card 0 port $PORT"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $PORT --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$PORT/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
MIXED_LAST=""
mixed() {
  local name=$1 tree=$2
  mixed_once $name $tree
  MIXED_LAST=$name
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  if [ -n "$p50" ] && [ "$p50" -le 649 ]; then
    log "$line, within 649"
  else
    log "$line, over 649: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree
    MIXED_LAST=${name}r
    log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O ${name}r)"
  fi
}
until ! pgrep -f "box/baseline.py .*--card 0 " > /dev/null; do sleep 20; done
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-one-context-visual-graph-read
git worktree add -q --detach $OCVG7 FETCH_HEAD
if [ "$(git -C $OCVG7 rev-parse --short=9 HEAD)" != "$OCVG7_SHA" ] || ! git -C $OCVG7 diff --quiet HEAD -- sglang_omni; then
  log "q1122_j stopped: ocvg7 is not a clean $OCVG7_SHA"
  exit 1
else :; fi
log "q1122_j start, $(node_free)"
alone js1 $OCVG7 1,16
alone js2 $BVG 1,16
alone js3 $BVG 1,16
alone js4 $OCVG7 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "part 2 final turn order (085064564) against its base, card 0 B A A B, speech alone $cell" \
    two=js2,js3 one=js1,js4 >> $O/queue.log 2>&1
done
python3 $K/first_audio_states.py --cell c16_events --main $O/js2,$O/js3 --branch $O/js1,$O/js4 > $O/j_states_c16.txt 2>&1
grep -E "server first audio|frames until|thinker prefill|thinker build|hop to the thinker|talker build" $O/j_states_c16.txt >> $O/queue.log
log "J1 read"
mixed jm1_b $OCVG7
b1=$MIXED_LAST
mixed jm2_a $BVG
a2=$MIXED_LAST
mixed jm3_a $BVG
a3=$MIXED_LAST
mixed jm4_b $OCVG7
b4=$MIXED_LAST
python3 $K/body_table.py $O c64 "part 2 final turn order against its base, card 0 B A A B, speech 8/s and video 1/s" \
  two=$a2,$a3 one=$b1,$b4 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O two=$a2,$a3 one=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "J2 read"
alone jc1 $OCVG7 64
alone jc2 $BVG 64
alone jc3 $BVG 64
alone jc4 $OCVG7 64
python3 $K/body_table.py $O c64 "part 2 final turn order against its base, card 0 B A A B, speech alone c64" \
  two=jc2,jc3 one=jc1,jc4 >> $O/queue.log 2>&1
log "J3 read"
seeded ji1_b $OCVG7
seeded ji2_a $BVG
log "identity part 2 $(python3 $K/wav_identity.py $O c1_clean ji2_a ji1_b 2>&1 | grep -m1 'shared')"
seeded ji3_b $IXC
seeded ji4_a $NMAIN
log "identity fix A $(python3 $K/wav_identity.py $O c1_clean ji4_a ji3_b 2>&1 | grep -m1 'shared')"
log "J4 read"
log "q1122_j done"
