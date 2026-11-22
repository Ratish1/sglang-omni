#!/usr/bin/env bash
# 11-23, node-0001, card 0 with card 1 idle, after q1123_g. The clean rerun of the beside-video arm block. q1123_e read
# all three arms (blocking-copies, no-stream-priority, read-in-turn) RTF +6.7 to +9.0 % and first audio mean +8 to
# +12 % against F, most of it the preprocessing queue (a CPU stage none of them changes), while card 1 ran E1's seeded
# c64 boots and two text Video-MME runs, the video ones preprocessing on the same host; F's boots, first and last, had
# card 1 mostly quiet. Three mechanisms moving together is the host, not the code. Here nothing runs on card 1, and E1
# (the full prefill graph) joins the symmetric order: F, BC, NSP, RIT, E1, E1, RIT, NSP, BC, F, gated at 649 ms.
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/ocvg7 E1=/workspace/p-full-prefill-64
BC=/workspace/x-blocking-copies NSP=/workspace/x-no-stream-priority RIT=/workspace/x-read-in-turn
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS0=0-55,112-167
PORT0=29470
GATE=649
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
LAST=""
mixed() {
  local name=$1 tree=$2
  mixed_once $name $tree
  LAST=$name
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  if [ -n "$p50" ] && [ "$p50" -le $GATE ]; then
    log "$line, within $GATE"
  else
    log "$line, over $GATE: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree
    LAST=${name}r
    log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O ${name}r)"
  fi
}
until grep -q "q1123_g done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py" > /dev/null && [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; do
  sleep 20
done
log "q1123_h start, card 1 idle, $(node_free)"
mixed eh1_f $F; f1=$LAST
mixed eh2_bc $BC; bc1=$LAST
mixed eh3_nsp $NSP; nsp1=$LAST
mixed eh4_rit $RIT; rit1=$LAST
mixed eh5_e1 $E1; e1a=$LAST
mixed eh6_e1 $E1; e1b=$LAST
mixed eh7_rit $RIT; rit2=$LAST
mixed eh8_nsp $NSP; nsp2=$LAST
mixed eh9_bc $BC; bc2=$LAST
mixed eh10_f $F; f2=$LAST
for pair in "blocking-copies $bc1,$bc2" "no-stream-priority $nsp1,$nsp2" "read-in-turn $rit1,$rit2" "E1 $e1a,$e1b"; do
  set -- $pair
  python3 $K/body_table.py $O c64 "$1 against F, 8/s beside video, card 0, card 1 idle, F first and last" F=$f1,$f2 $1=$2 \
    >> $O/queue.log 2>&1
  python3 $K/first_audio_states.py --cell c64_events --main $O/$f1,$O/$f2 --branch $O/${2%,*},$O/${2#*,} \
    > $O/h_states_$1.txt 2>&1
  grep -E "server first audio|preprocessing queue|thinker prefill|thinker chunks|frames until" $O/h_states_$1.txt \
    >> $O/queue.log
done
python3 $K/mixed_cell_load.py $O $f1 $f2 $bc1 $bc2 $nsp1 $nsp2 $rit1 $rit2 $e1a $e1b >> $O/queue.log 2>&1
log "H read"
log "q1123_h done"
