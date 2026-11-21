#!/usr/bin/env bash
# 11-21, node-0001, card 0 after q1121_stackoc. Part 2 alone on its landing base: the combined form (OCVG 20d57c8b4,
# tree 3fdf8496fd26: both changes, part 1, the index copy fix, part 2) against the same without part 2 in two processes
# (BVG 6aff09f46, tree 32cb9d81e911). Card 0 alone, B A A B, speech 8/s seed 3451 beside video 1/s seed 7 (cap 8),
# every video boot gated; then speech alone c1 and c16 B A A B. Then the index copy fix alone on main (IXC 4ca23252e,
# tree 03e7268c8857, against NMAIN e0482a08c), speech beside video B A A B. Then, with every measurement done, the unit
# suites the stack touches on the combined and stack heads (they run kernels on card 0).
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
OCVG=/workspace/ocvg BVG=/workspace/bvg SOCVG=/workspace/socvg IXC=/workspace/ixc NMAIN=/workspace/nmain
OCVG_SHA=20d57c8b4 BVG_SHA=6aff09f46 SOCVG_SHA=26224b691 IXC_SHA=4ca23252e NMAIN_SHA=e0482a08c
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS=0-55,112-167
UNIT_SUITES="tests/unit_test/model_runner tests/unit_test/qwen3_omni tests/unit_test/pipeline tests/unit_test/preprocessing tests/unit_test/relay tests/unit_test/scheduling"
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
  local name=$1 tree=$2 limit=$3
  mixed_once $name $tree
  LAST=$name
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  if [ -n "$p50" ] && [ "$p50" -le $limit ]; then
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
units() {
  local name=$1 tree=$2
  mkdir -p $O/units_$name
  cd $tree
  PYTHONPATH=$tree taskset -c $CPUS python3 -m pytest -q -p no:cacheprovider $UNIT_SUITES > $O/units_$name/units.log 2>&1
  log "units $name $(git -C $tree rev-parse --short HEAD) rc $? $(grep -E 'passed|failed' $O/units_$name/units.log | tail -1)"
}
until grep -q "q1121_stackoc done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 0 " > /dev/null; do sleep 20; done
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-both-changes-visual-graph
git worktree add -q --detach $BVG FETCH_HEAD
for pair in "$OCVG $OCVG_SHA" "$BVG $BVG_SHA" "$SOCVG $SOCVG_SHA" "$IXC $IXC_SHA" "$NMAIN $NMAIN_SHA"; do
  set -- $pair
  if [ "$(git -C $1 rev-parse --short=9 HEAD)" != "$2" ] || ! git -C $1 diff --quiet HEAD -- sglang_omni; then
    log "q1121_p2base stopped: $1 is not a clean $2"
    exit 1
  else :; fi
done
log "tree bvg $(git -C $BVG rev-parse --short HEAD) $(git -C $BVG rev-parse --short=12 HEAD:sglang_omni)"
log "q1121_p2base start, $(node_free)"
mixed pb1_b $OCVG 649
b1=$LAST
mixed pb2_a $BVG 649
a2=$LAST
mixed pb3_a $BVG 649
a3=$LAST
mixed pb4_b $OCVG 649
b4=$LAST
python3 $K/body_table.py $O c64 "part 2 on its landing base: one context (85ba60a52) against two processes (ed2980ce3), card 0 B A A B, speech 8/s and video 1/s" \
  two=$a2,$a3 one=$b1,$b4 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O two=$a2,$a3 one=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "pb read"
alone ps1_b $OCVG 1,16
alone ps2_a $BVG 1,16
alone ps3_a $BVG 1,16
alone ps4_b $OCVG 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "part 2 on its landing base, card 0 B A A B, speech alone $cell" \
    two=ps2_a,ps3_a one=ps1_b,ps4_b >> $O/queue.log 2>&1
done
log "ps read"
mixed ix1_b $IXC 649
b1=$LAST
mixed ix2_a $NMAIN 649
a2=$LAST
mixed ix3_a $NMAIN 649
a3=$LAST
mixed ix4_b $IXC 649
b4=$LAST
python3 $K/body_table.py $O c64 "index copy fix alone (4ca23252e) against main (e0482a08c), card 0 B A A B, speech 8/s and video 1/s" \
  main=$a2,$a3 fix=$b1,$b4 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O main=$a2,$a3 fix=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "ix read"
units ocvg $OCVG
units socvg $SOCVG
log "q1121_p2base done"
