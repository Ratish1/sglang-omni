#!/usr/bin/env bash
# 11-28, node-0001, after q1128_b: the serving cost of each knob that answer invariance needs (q1128_b: the split pin and
# mixed chunk off are needed, eager graphs are not, the radix cache only moves answers by boot), on main alone, graphs on,
# radix on, same-time pairs on the two cards, swapped:
#   D  main as served
#   P  main with FlashAttention split-KV pinned at 1 and the scheduler metadata dropped (thinker_numerics_probe, every
#      stage process of the server, so the talker's attention is pinned too)
#   X  main with the thinker's mixed chunk off
# Cells: speech 8/s beside video 1/s colocated (D P, P D, D X, X D), speech alone c16 and c64 (same order), text Video-MME
# c16 once a boot for speed (same order).
set -u
O=/data/v1128
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
MIXED_OFF="--thinker.engine.enable_mixed_chunk false"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
port_of() { echo $((20000 + 4000 * $1 + $2)); }
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
knob_flags() { case $1 in x) echo "$MIXED_OFF" ;; *) echo "" ;; esac; }
knob_env() {
  case $1 in
    p) echo "PYTHONPATH=$K/thinker_numerics_probe:$M OMNI_NUMERICS_PROBE=$O/$2/num OMNI_NUMERICS_TARGET_LENS=1 OMNI_FA_SPLITS=1" ;;
    *) echo "PYTHONPATH=$M" ;;
  esac
}
log "q1128_c start"
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
video_loop() {
  local name=$1 driver=$2 port=$3 cpus=$4
  cd $M
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$M taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
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
mixed() {
  local name=$1 knob=$2 card=$3 offset=$4
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  mkdir -p $O/$name
  note $name $M "colocated, speech 8/s beside video 1/s, knob $knob, card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $M --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K $(knob_flags $knob)" --serve-env "$(knob_env $knob $name)" --omni-chunk-audio \
    "--bench-extra=--request-rate=8" --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $driver $port $cpus &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}
speech() {
  local name=$1 knob=$2 card=$3 offset=$4
  mkdir -p $O/$name
  note $name $M "colocated, speech alone c16 and c64, knob $knob, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $M --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16,64 --serve-extra "$CONTEXT_32K $(knob_flags $knob)" \
    --serve-env "$(knob_env $knob $name)" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video() {
  local name=$1 knob=$2 card=$3 offset=$4
  mkdir -p $O/$name
  note $name $M "CI speech pipeline, text Video-MME c16 once, knob $knob, card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $M --model omni_ci_onecard --card $card \
    --port $(port_of $card $offset) --out $O/$name --serve-extra "$(knob_flags $knob)" \
    --serve-env "$(knob_env $knob $name)" --bench videomme_text --loads 16 --passes clean > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
offset=100
for cell in mixed speech video; do
  offset=$((offset + 200))
  both "$cell k${cell}_d1 d 0 $offset" "$cell k${cell}_p1 p 1 $offset"
  offset=$((offset + 200))
  both "$cell k${cell}_p2 p 0 $offset" "$cell k${cell}_d2 d 1 $offset"
  offset=$((offset + 200))
  both "$cell k${cell}_d3 d 0 $offset" "$cell k${cell}_x1 x 1 $offset"
  offset=$((offset + 200))
  both "$cell k${cell}_x2 x 0 $offset" "$cell k${cell}_d4 d 1 $offset"
done
python3 $K/body_table.py $O c64 "split pin against main, 8/s beside video 1/s" main=kmixed_d1,kmixed_d2 \
  knob=kmixed_p1,kmixed_p2 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "mixed chunk off against main, 8/s beside video 1/s" main=kmixed_d3,kmixed_d4 \
  knob=kmixed_x1,kmixed_x2 >> $O/queue.log 2>&1
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "split pin against main, speech alone, $cell" main=kspeech_d1,kspeech_d2 \
    knob=kspeech_p1,kspeech_p2 >> $O/queue.log 2>&1
  python3 $K/body_table.py $O $cell "mixed chunk off against main, speech alone, $cell" main=kspeech_d3,kspeech_d4 \
    knob=kspeech_x1,kspeech_x2 >> $O/queue.log 2>&1
done
for name in kvideo_d1 kvideo_p1 kvideo_p2 kvideo_d2 kvideo_d3 kvideo_x1 kvideo_x2 kvideo_d4; do
  log "video $name $(grep -h -m1 -o '"throughput_qps": [0-9.]*' $O/$name/c16_clean/bench/*.json 2>/dev/null | head -1) $(grep -h -m1 -o '"latency_mean_s": [0-9.]*' $O/$name/c16_clean/bench/*.json 2>/dev/null | head -1)"
done
log "q1128_c done"
