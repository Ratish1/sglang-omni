#!/usr/bin/env bash
# 11-21, node-0001, after q1121_v2's v3a pair. NUMA node 1 holds 6 GiB free against 152 on node 0 (942 GiB of other
# tenants' page cache): a fresh host page on socket 1 costs 2.3 times socket 0 (2 GiB zero-fill and clone 638 to 688 ms
# against 281, binding to node 1 does not help), and every card 1 boot since 18:13 ran its video preprocessing 55 to 70 %
# slower (890, 983 ms against 569 to 593). Same-time pairs across the two sockets are no longer equal, so each
# comparison runs on card 0 alone, four boots in A B B A order (pairs 1-2 and 4-3), each video boot gated on its video
# preprocessing median (main-based trees 580 ms, stack-based 445 ms; over 12 % above is rerun once). Speech 8/s seed 3451
# beside video 1/s seed 7 (cap 8) unless noted.
#   W1  the receive-only shared-memory change (PSHM2) against main
#   W2  the stack with both changes (NSBOTH2) against main
#   W3  CUDA traces, main and the graph change (PFIX), one boot each: talker steps and thinker eager steps
#   W4  both changes (PBOTH2) against main
#   W5  the graph change on the receive-only base: PBOTH2 against PSHM2
#   W6  the stack with both changes against main, beside the looping Video-MME client (4 in flight)
#   W7  both changes against main, speech alone c1 and c16
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain PFIX=/workspace/pfix NSTACK=/workspace/nstack
PSHM2=/workspace/pshm2 PBOTH2=/workspace/pboth2 NSBOTH2=/workspace/nsboth2
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
  echo $((28000 + 160 + 200 * (BOOTS % 20)))
}
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
}
expected_ms() { case $1 in $NSTACK | $NSBOTH2) echo 445 ;; *) echo 580 ;; esac; }
video_loop() {
  local name=$1 tree=$2 port=$3 driver=$4 video=$5
  local video_args=(--max-concurrency 8 --request-rate 1)
  if [ $video = loop ]; then video_args=(--max-concurrency 4); else :; fi
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 "${video_args[@]}" \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
}
mixed_once() {
  local name=$1 tree=$2 kind=$3 video=$4
  local port=$(next_port)
  local pass_args=(--passes warm,events)
  if [ $kind = nsys ]; then
    pass_args=(--passes warm,nsys --nsys-session $name --nsys-trace cuda --nsys-graph-trace graph --nsys-delay 20
      --nsys-seconds 15)
  else :; fi
  quiet_host
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video $video card 0 $kind port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" "${pass_args[@]}" \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $port $driver $video &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
mixed() {
  local name=$1 tree=$2 kind=$3 video=$4
  mixed_once $name $tree $kind $video
  LAST=$name
  if [ $kind = nsys ]; then return; else :; fi
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  local limit=$(($(expected_ms $tree) * 112 / 100))
  if [ -n "$p50" ] && [ "$p50" -le "$limit" ]; then
    log "$line, within $limit"
  else
    log "$line, over $limit: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree $kind $video
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
  LAST=$name
}
abba() {
  local tag=$1 base=$2 change=$3 video=$4 title=$5
  mixed ${tag}1_a $base events $video
  local a1=$LAST
  mixed ${tag}2_b $change events $video
  local b2=$LAST
  mixed ${tag}3_b $change events $video
  local b3=$LAST
  mixed ${tag}4_a $base events $video
  local a4=$LAST
  python3 $K/body_table.py $O c64 "$title" base=$a1,$a4 change=$b2,$b3 >> $O/queue.log 2>&1
  python3 $K/first_chunk_gap.py --cell c64_events $O base=$a1,$a4 change=$b2,$b3 2>&1 | grep gaps >> $O/queue.log
  log "$tag read"
}
until ! pgrep -f "kit/tasks/runtime-perf-research/box/baseline.py" > /dev/null; do sleep 20; done
log "q1121_v4 start, $(node_free)"
abba w1 $NMAIN $PSHM2 rate "receive-only shm against main, card 0 A B B A, speech 8/s and video 1/s"
abba w2 $NMAIN $NSBOTH2 rate "stack with both changes against main, card 0 A B B A, speech 8/s and video 1/s"
mixed w3_m $NMAIN nsys rate
mixed w3_p $PFIX nsys rate
for name in w3_m w3_p; do
  (cd $O/$name && python3 - c64_nsys/trace.sqlite serve.log < $K/talker_steps.py > talker_steps.txt 2>&1
    python3 - c64_nsys/trace.sqlite serve.log < $K/thinker_steps.py > thinker_steps.txt 2>&1)
  log "trace read $name: $(grep -c . $O/$name/talker_steps.txt) talker lines, $(grep -c . $O/$name/thinker_steps.txt) thinker lines"
done
log "w3 read"
abba w4 $NMAIN $PBOTH2 rate "both changes against main, card 0 A B B A, speech 8/s and video 1/s"
abba w5 $PSHM2 $PBOTH2 rate "graph change on the receive-only base, card 0 A B B A, speech 8/s and video 1/s"
abba w6 $NMAIN $NSBOTH2 loop "stack with both changes against main, card 0 A B B A, 8/s beside 4 looping videos"
alone w7_1a $NMAIN 1,16
alone w7_2b $PBOTH2 1,16
alone w7_3b $PBOTH2 1,16
alone w7_4a $NMAIN 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "both changes against main, card 0 A B B A, speech alone $cell" \
    main=w7_1a,w7_4a both=w7_2b,w7_3b >> $O/queue.log 2>&1
done
log "w7 read"
log "q1121_v4 done"
