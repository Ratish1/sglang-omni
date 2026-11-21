#!/usr/bin/env bash
# 11-21, node-0001, card 0. The one-context prototype (ONECTX 8f3ede3ef, tree 119ca65a208f) against the tree it is built
# on (PBOTH2, both changes on main), card 0 alone, A B B A, speech 8/s seed 3451 beside video 1/s seed 7 (cap 8), each
# video boot gated on its video preprocessing median (580 ms, over 12 % reruns once). q1121_v7 stops after its W5
# boot in flight; its W5, W7 and W6 run again after this. Then one CUDA trace with context switches of each arm.
set -u
O=/data/v1121
K=/workspace/kit/tasks/runtime-perf-research/box
PBOTH2=/workspace/pboth2 ONECTX=/workspace/onectx
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
  echo $((28000 + 190 + 200 * (BOOTS % 20)))
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
  local name=$1 tree=$2 kind=$3
  local port=$(next_port)
  local pass_args=(--passes warm,events)
  if [ $kind = nsys ]; then
    pass_args=(--passes warm,nsys,events --nsys-session $name --nsys-trace cuda --nsys-graph-trace graph --nsys-gpuctxsw
      --nsys-delay 20 --nsys-seconds 15)
  else :; fi
  quiet_host
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s card 0 $kind port $port"
  taskset -c $CPUS python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio "--bench-extra=--request-rate=8" "${pass_args[@]}" \
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
  local name=$1 tree=$2 kind=$3
  mixed_once $name $tree $kind
  LAST=$name
  local line=$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)
  local p50=$(echo "$line" | sed -n 's/.*preprocess p50 \([0-9]*\) ms.*/\1/p')
  if [ -n "$p50" ] && [ "$p50" -le 649 ]; then
    log "$line, within 649"
  else
    log "$line, over 649: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree $kind
    LAST=${name}r
    log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O ${name}r)"
  fi
}
until grep -q "^boot w51_a .* start" $O/queue.log 2>/dev/null; do sleep 10; done
pkill -f "^bash /data/v1121/q1121_v7.sh"
log "q1121_v7 stopped during its W5 first boot (it runs on); W5, W7 and W6 run after the one-context comparison"
until ! pgrep -f "box/baseline.py .*--card 0 " > /dev/null; do sleep 20; done
log "q1121_oc2 start, $(node_free)"
mixed oa1_a $PBOTH2 events
a1=$LAST
mixed oa2_b $ONECTX events
b2=$LAST
mixed oa3_b $ONECTX events
b3=$LAST
mixed oa4_a $PBOTH2 events
a4=$LAST
python3 $K/body_table.py $O c64 "one context against two processes (both changes), card 0 A B B A, speech 8/s and video 1/s" \
  two=$a1,$a4 one=$b2,$b3 >> $O/queue.log 2>&1
python3 $K/first_chunk_gap.py --cell c64_events $O two=$a1,$a4 one=$b2,$b3 2>&1 | grep gaps >> $O/queue.log
python3 $K/mixed_cell_load.py $O $b2 $b3 $a1 $a4 >> $O/queue.log 2>&1
log "oa read"
mixed ot1_b $ONECTX nsys
(cd $O/ot1_b && python3 $K/ctxsw_read.py c64_nsys/trace.sqlite > ctxsw.txt 2>&1)
log "trace read ot1_b: $(grep -c . $O/ot1_b/ctxsw.txt) ctxsw lines"
log "q1121_oc2 done"
