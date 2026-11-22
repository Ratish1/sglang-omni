#!/usr/bin/env bash
# 11-23, node-0001, after q1123_a. Fix A's other cells on its landing base (upstream main 22bf41ef5, which carries
# #3936's streamed audio output changes since the e0482a08c pairs): fix A 3491dc8a7 against main.
#   card 0: speech 8/s seed 3451 beside video 1/s (8 in flight), gated at 649 ms preprocessing p50, B A A B
#   card 1: speech alone c16, B A A B, then the seeded c1 wav identity (main, then fix A)
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
UMAIN=/workspace/umain FIXA=/workspace/fixa
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS0=0-55,112-167 CPUS1=56-111,168-223
PORT0=28370 PORT1=30370
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
video_loop() {
  local name=$1 tree=$2 driver=$3
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $CPUS0 python3 $K/videomme_speed.py --host 127.0.0.1 --port $PORT0 \
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
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s card 0 port $PORT0"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $PORT0 --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" --passes warm,events \
    > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$PORT0/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
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
alone() {
  local name=$1 tree=$2
  note $name $tree "seedtts c16 alone card 1 port $PORT1"
  taskset -c $CPUS1 python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $PORT1 --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
seeded() {
  local name=$1 tree=$2
  note $name $tree "seeded c1 card 1 port $PORT1"
  taskset -c $CPUS1 python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $PORT1 --out $O/$name \
    --loads 1 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes clean "--bench-extra=--seed 3451" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
card0() {
  mixed am1_f $FIXA
  local m1=$MIXED_LAST
  mixed am2_m $UMAIN
  local m2=$MIXED_LAST
  mixed am3_m $UMAIN
  local m3=$MIXED_LAST
  mixed am4_f $FIXA
  local m4=$MIXED_LAST
  python3 $K/body_table.py $O c64 "fix A against main, 8/s beside video, card 0 B A A B" main=$m2,$m3 fix=$m1,$m4 \
    >> $O/queue.log 2>&1
  python3 $K/mixed_cell_load.py $O $m1 $m4 $m2 $m3 >> $O/queue.log 2>&1
  python3 $K/first_audio_states.py --cell c64_events --main $O/$m2,$O/$m3 --branch $O/$m1,$O/$m4 \
    > $O/a2_states_mixed.txt 2>&1
  grep -E "server first audio|preprocessing queue|thinker prefill|frames until" $O/a2_states_mixed.txt >> $O/queue.log
  log "A2 mixed read"
}
card1() {
  alone as1_f $FIXA
  alone as2_m $UMAIN
  alone as3_m $UMAIN
  alone as4_f $FIXA
  python3 $K/body_table.py $O c16 "fix A against main, speech alone c16, card 1 B A A B" main=as2_m,as3_m \
    fix=as1_f,as4_f >> $O/queue.log 2>&1
  log "A2 c16 read"
  seeded ai1_m $UMAIN
  seeded ai2_f $FIXA
  log "identity fix A on 22bf41ef5 $(python3 $K/wav_identity.py $O c1_clean ai1_m ai2_f 2>&1 | grep -m1 'shared')"
}
until grep -q "q1123_a done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_a2 start, $(node_free)"
card0 &
c0=$!
card1 &
c1=$!
wait $c0 $c1
log "q1123_a2 done"
