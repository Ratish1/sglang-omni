#!/usr/bin/env bash
# 11-23, node-0001, after q1123_h. Part 2 with the full prefill graph (E1 5ee1c55f2 on F) against part 2's base BVG
# (6aff09f46), the comparison part 2 is judged by. F against BVG read c16 inter chunk p99 +3.0 / +5.2 % (11-22), and E1
# against F reads c16 inter chunk p99 -2.4 / -10.7 %, every c16 row better in both pairs (q1123_g).
#   both cards: speech alone c1, then c16 and c64, same-time pairs swapped
#   card 0, card 1 idle: speech 8/s beside video 1/s, E1 BVG BVG E1, gated at 649 ms
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
E1=/workspace/p-full-prefill-64 BVG=/workspace/bvg
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS0=0-55,112-167 CPUS1=56-111,168-223
PORT0=29670
GATE=649
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
node_free() { numactl --hardware | awk '/free:/ {printf "node%s %s MB ", $2, $4}'; }
cpus_of() { if [ $1 = 1 ]; then echo $CPUS1; else echo $CPUS0; fi; }
port_of() { echo $((28000 + 2000 * $1 + $2)); }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
speech() {
  local name=$1 tree=$2 card=$3 offset=$4 loads=$5
  note $name $tree "c$loads card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads $loads --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
export OMNI_ARRIVAL_SEED=3451
export OMNI_VIDEO_ARRIVAL_SEED=7
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
until grep -q "q1123_h done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_i start"
quiet_host
both "speech ia1_b0 $BVG 0 1210 1" "speech ia1_e1 $E1 1 1210 1"
quiet_host
both "speech ia2_e0 $E1 0 1230 1" "speech ia2_b1 $BVG 1 1230 1"
python3 $K/body_table.py $O c1 "E1 against part 2's base, c1, same-time pairs swapped" base=ia1_b0,ia2_b1 E1=ia1_e1,ia2_e0 \
  >> $O/queue.log 2>&1
quiet_host
both "speech ib1_b0 $BVG 0 1250 16,64" "speech ib1_e1 $E1 1 1250 16,64"
quiet_host
both "speech ib2_e0 $E1 0 1270 16,64" "speech ib2_b1 $BVG 1 1270 16,64"
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "E1 against part 2's base, $cell, same-time pairs swapped" base=ib1_b0,ib2_b1 \
    E1=ib1_e1,ib2_e0 >> $O/queue.log 2>&1
done
python3 $K/first_chunk_gap.py --cell c16_events $O base=ib1_b0,ib2_b1 E1=ib1_e1,ib2_e0 2>&1 | grep gaps >> $O/queue.log
python3 $K/first_audio_states.py --cell c16_events --main $O/ib1_b0,$O/ib2_b1 --branch $O/ib1_e1,$O/ib2_e0 \
  > $O/i_states_c16.txt 2>&1
grep -E "server first audio|thinker prefill|thinker chunks|frames until" $O/i_states_c16.txt >> $O/queue.log
log "I speech read"
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
mixed im1_e $E1; m1=$LAST
mixed im2_b $BVG; m2=$LAST
mixed im3_b $BVG; m3=$LAST
mixed im4_e $E1; m4=$LAST
python3 $K/body_table.py $O c64 "E1 against part 2's base, 8/s beside video, card 0 E1 B B E1, card 1 idle" base=$m2,$m3 \
  E1=$m1,$m4 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O $m1 $m4 $m2 $m3 >> $O/queue.log 2>&1
python3 $K/first_audio_states.py --cell c64_events --main $O/$m2,$O/$m3 --branch $O/$m1,$O/$m4 > $O/i_states_mixed.txt 2>&1
grep -E "server first audio|preprocessing queue|thinker prefill|thinker chunks|frames until" $O/i_states_mixed.txt \
  >> $O/queue.log
log "I mixed read"
log "q1123_i done"
