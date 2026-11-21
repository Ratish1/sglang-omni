#!/usr/bin/env bash
# 11-21, node-0001, card 0, after q1121_oc2. The one-context form b9b7231f5 (tree fff21522a742: fair turn handed to the
# talker first, thinker yields at loop phases, thinker decodes synchronously in a shared process, its step-end wait
# outside the turn) read one pair against two processes (oa1_a, oa3_b): inter chunk mean -24.6 %, p99 -19.5 %, latency
# median -27.7 %, first audio -19.5 %, video -18 % s per 100 tokens. Now a clean B A A B of that exact form against
# the tree it is built on (PBOTH2), speech 8/s seed 3451 beside video 1/s seed 7 (cap 8), every video boot gated; then
# speech alone c1 and c16 in B A A B.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
PBOTH2=/workspace/pboth2 ONECTX=/workspace/onectx
ONECTX_SHA=b9b7231f5909b819bbdcdb2fbd62b5f858ce7536
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
  echo $((28000 + 170 + 200 * (BOOTS % 20)))
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
until grep -q "q1121_oc2 done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 0 " > /dev/null; do sleep 20; done
if [ "$(git -C $ONECTX rev-parse HEAD)" != "$ONECTX_SHA" ]; then
  log "q1121_oc3 stopped: $ONECTX is at $(git -C $ONECTX rev-parse --short HEAD), not ${ONECTX_SHA:0:9}"
  exit 1
else :; fi
log "q1121_oc3 start, $(node_free)"
mixed ob1_b $ONECTX
b1=$LAST
mixed ob2_a $PBOTH2
a2=$LAST
mixed ob3_a $PBOTH2
a3=$LAST
mixed ob4_b $ONECTX
b4=$LAST
python3 $K/body_table.py $O c64 "one context b9b7231f5 against two processes (both changes), card 0 B A A B, speech 8/s and video 1/s" \
  two=$a2,$a3 one=$b1,$b4 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O two=$a2,$a3 one=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "ob read"
alone os1_b $ONECTX 1,16
alone os2_a $PBOTH2 1,16
alone os3_a $PBOTH2 1,16
alone os4_b $ONECTX 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "one context against two processes, card 0 B A A B, speech alone $cell" \
    two=os2_a,os3_a one=os1_b,os4_b >> $O/queue.log 2>&1
done
log "os read"
log "q1121_oc3 done"
