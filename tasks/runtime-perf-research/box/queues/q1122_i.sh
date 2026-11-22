#!/usr/bin/env bash
# 11-22, node-0001. At c64 the turn rules hurt part 2 (QPS -0.7 %, inter chunk p99 +13.7 %, first audio mean +1.3 %
# against its base): the talker is the bottleneck there (talker build and queue about 1,060 ms of first audio) and the
# thinker, always holding a speaking request before its first emit, took its turns. Now a talker with requests still
# waiting after its selection is expedited and never defers (d47b9edc2), so under talker-bound load the order is
# talker-first as before the rules. Combined: OCVG6 b60cac26c (tree c5f8e02ca6f7) against part 2's base BVG 6aff09f46.
#   I1 (card 0)  speech alone c64, B A A B
#   I2 (card 0)  speech 8/s beside video 1/s, B A A B, every video boot gated
#   I3 (card 0)  speech alone c1 and c16, B A A B
#   I4 (card 0)  seeded c1 wav identity: OCVG6 against BVG, and the index copy fix (IXC 4ca23252e) against main
#   I5 (card 1, after q1122_g)  WER and SIM of I1's cells
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain BVG=/workspace/bvg OCVG6=/workspace/ocvg6 IXC=/workspace/ixc
OCVG6_SHA=b60cac26c
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS0=0-55,112-167
ASR_PORT=30620
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
card1_used_mib() { nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
alone() {
  local name=$1 tree=$2 loads=$3
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts loads $loads alone card 0 port 28530"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28530 --out $O/$name \
    --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
seeded() {
  local name=$1 tree=$2
  note $name $tree "seeded c1 card 0 port 28530"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28530 --out $O/$name \
    --loads 1 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes clean "--bench-extra=--seed 3451" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video_loop() {
  local name=$1 tree=$2 port=$3 driver=$4
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS0 python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
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
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s card 0 port 28530"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port 28530 --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:28530/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree 28530 $driver &
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
git worktree add -q --detach $OCVG6 FETCH_HEAD
if [ "$(git -C $OCVG6 rev-parse --short=9 HEAD)" != "$OCVG6_SHA" ] || ! git -C $OCVG6 diff --quiet HEAD -- sglang_omni; then
  log "q1122_i stopped: ocvg6 is not a clean $OCVG6_SHA"
  exit 1
else :; fi
log "q1122_i start, $(node_free)"
alone ic1 $OCVG6 64
alone ic2 $BVG 64
alone ic3 $BVG 64
alone ic4 $OCVG6 64
python3 $K/body_table.py $O c64 "part 2 with the backlog rule (b60cac26c) against its base, card 0 B A A B, speech alone c64" \
  two=ic2,ic3 one=ic1,ic4 >> $O/queue.log 2>&1
log "I1 read"
(
  until grep -q "^q1122_g done" $O/queue.log 2>/dev/null; do sleep 20; done
  until [ "$(card1_used_mib)" -lt 1024 ]; do sleep 5; done
  cd $NMAIN
  setsid python3 -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $ASR_PORT --asr.gpu 1 \
    > $O/asr_i_serve.log 2>&1 &
  echo $! > $O/asr_i.pgid
  until curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null || ! kill -0 $(cat $O/asr_i.pgid) 2>/dev/null; do
    sleep 5
  done
  for boot in ic1 ic2 ic3 ic4; do
    cell=$O/$boot/c64_events
    python3 -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --lang en --voice-clone --port $ASR_PORT \
      --asr-concurrency 64 --output-dir $cell/bench > $cell/transcribe.log 2>&1
    log "wer $boot rc $?"
  done
  kill -TERM -- -$(cat $O/asr_i.pgid) 2>/dev/null
  until [ "$(card1_used_mib)" -lt 1024 ]; do sleep 5; done
  for boot in ic1 ic2 ic3 ic4; do
    cell=$O/$boot/c64_events
    python3 -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --lang en --voice-clone --device cuda:1 \
      --output-dir $cell/bench > $cell/similarity.log 2>&1
    log "similarity $boot rc $? $(grep -m1 -o 'Tried to allocate [0-9.]* [GM]iB' $cell/similarity.log)"
  done
  python3 $K/wer_sim_cells.py backlog_c64_1=$O/ic1/c64_events/bench base_c64_2=$O/ic2/c64_events/bench \
    base_c64_3=$O/ic3/c64_events/bench backlog_c64_4=$O/ic4/c64_events/bench >> $O/queue.log 2>&1
  log "I5 read"
) &
scoring=$!
mixed im1_b $OCVG6
b1=$MIXED_LAST
mixed im2_a $BVG
a2=$MIXED_LAST
mixed im3_a $BVG
a3=$MIXED_LAST
mixed im4_b $OCVG6
b4=$MIXED_LAST
python3 $K/body_table.py $O c64 "part 2 with the backlog rule against its base, card 0 B A A B, speech 8/s and video 1/s" \
  two=$a2,$a3 one=$b1,$b4 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O two=$a2,$a3 one=$b1,$b4 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b1 $b4 $a2 $a3 >> $O/queue.log 2>&1
log "I2 read"
alone is1 $OCVG6 1,16
alone is2 $BVG 1,16
alone is3 $BVG 1,16
alone is4 $OCVG6 1,16
for cell in c1 c16; do
  python3 $K/body_table.py $O $cell "part 2 with the backlog rule against its base, card 0 B A A B, speech alone $cell" \
    two=is2,is3 one=is1,is4 >> $O/queue.log 2>&1
done
python3 $K/first_audio_states.py --cell c16_events --main $O/is2,$O/is3 --branch $O/is1,$O/is4 > $O/i_states_c16.txt 2>&1
grep -E "server first audio|frames until|thinker prefill|thinker build|hop to the thinker|talker build" $O/i_states_c16.txt >> $O/queue.log
log "I3 read"
seeded ii1_b $OCVG6
seeded ii2_a $BVG
log "identity part 2 $(python3 $K/wav_identity.py $O c1_clean ii2_a ii1_b 2>&1 | grep -m1 'shared')"
seeded ii3_b $IXC
seeded ii4_a $NMAIN
log "identity fix A $(python3 $K/wav_identity.py $O c1_clean ii4_a ii3_b 2>&1 | grep -m1 'shared')"
log "I4 read"
wait $scoring
log "q1122_i done"
