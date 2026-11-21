#!/usr/bin/env bash
# 11-21, node-0001, card 0, after q1121_v6's context switch traces. The traces show the talker's slower steps under
# either change are card arbitration (main against the transport change: talker share of the card 47.6 against 49.2 %,
# switches 807 against 800 a second, other processes' device time inside a talker decode step 5.5 against 5.9 ms, the
# talker's end-of-step wait p50 5.48 against 6.06 ms), not a defect inside either change. Validation of the current
# heads goes on, card 0 alone, A B B A, each video boot gated (NUMA node 1 back to 6 GiB free):
#   W2b  a fresh B A pair for W2 (its fourth boot ran without video)
#   W4   both changes (PBOTH2) against main
#   W5   the graph change on the receive-only base: PBOTH2 against PSHM2
#   W7   both changes against main, speech alone c1 and c16
#   W6   the stack with both changes against main, beside the looping Video-MME client (4 in flight)
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain NSTACK=/workspace/nstack
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
  echo $((28000 + 180 + 200 * (BOOTS % 20)))
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
  local name=$1 tree=$2 video=$3
  local port=$(next_port)
  quiet_host
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video $video card 0 port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
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
  local name=$1 tree=$2 video=$3
  mixed_once $name $tree $video
  LAST=$name
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  local limit=$(($(expected_ms $tree) * 112 / 100))
  if [ -n "$p50" ] && [ "$p50" -le "$limit" ]; then
    log "$line, within $limit"
  else
    log "$line, over $limit: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree $video
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
  mixed ${tag}1_a $base $video
  local a1=$LAST
  mixed ${tag}2_b $change $video
  local b2=$LAST
  mixed ${tag}3_b $change $video
  local b3=$LAST
  mixed ${tag}4_a $base $video
  local a4=$LAST
  python3 $K/body_table.py $O c64 "$title" base=$a1,$a4 change=$b2,$b3 >> $O/queue.log 2>&1
  python3 $K/first_chunk_gap.py --cell c64_events $O base=$a1,$a4 change=$b2,$b3 2>&1 | grep gaps >> $O/queue.log
  log "$tag read"
}
until grep -q "q1121_v6 done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "kit/tasks/runtime-perf-research/box/baseline.py" > /dev/null; do sleep 20; done
log "q1121_v7 start, $(node_free)"
mixed w25_b $NSBOTH2 rate
b5=$LAST
mixed w26_a $NMAIN rate
a6=$LAST
python3 $K/body_table.py $O c64 "stack with both changes against main, card 0, W2 with its fresh B A pair" \
  main=w21_a,$a6 both=w22_b,w23_b,$b5 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O main=w21_a,$a6 both=w22_b,w23_b,$b5 2>&1 | grep gaps >> $O/queue.log
log "w2b read"
abba w4 $NMAIN $PBOTH2 rate "both changes against main, card 0 A B B A, speech 8/s and video 1/s"
abba w5 $PSHM2 $PBOTH2 rate "graph change on the receive-only base, card 0 A B B A, speech 8/s and video 1/s"
alone w7_1a $NMAIN 1,16
alone w7_2b $PBOTH2 1,16
alone w7_3b $PBOTH2 1,16
alone w7_4a $NMAIN 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "both changes against main, card 0 A B B A, speech alone $cell" \
    main=w7_1a,w7_4a both=w7_2b,w7_3b >> $O/queue.log 2>&1
done
log "w7 read"
abba w6 $NMAIN $NSBOTH2 loop "stack with both changes against main, card 0 A B B A, 8/s beside 4 looping videos"
log "q1121_v7 done"
