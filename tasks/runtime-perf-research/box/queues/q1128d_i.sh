#!/usr/bin/env bash
# 11-28 second lease, node-0001, after q1128d_h: the running video limit against speech underruns on a saturated shared
# card. Under the H20 emulation, speech c16 closed loop beside 8 looping videos had 14 to 45 underrun streams a pass on
# the PR at limit 8 against main's 0 and 2: the lane pushes more video prefill through the shared thinker (8,192-token
# steps of about 230 ms) and realtime speech misses its deadlines. DEV2 (4cac082f9) at limits 2 and 4 against MAIN, same
# emulation and load, then limit 2 against main on the H200 without a ballast (q1128d_b's cached loop).
set -u
O=/data/v1128b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/main V=/workspace/dev2
H20_FREE_MIB=97871
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
export OMNI_ARRIVAL_SEED=3451
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
video_loop() {
  local name=$1 tree=$2 driver=$3 port=$4 cpus=$5
  cd $tree
  pass=0
  while kill -0 $driver 2>/dev/null; do
    PYTHONPATH=$tree taskset -c $cpus python3 $K/videomme_speed.py --host 127.0.0.1 --port $port \
      --repo-id zhaochenyang20/Video_MME_ci --video-fps 2 --video-max-frames 128 --video-max-pixels 401408 \
      --max-tokens 256 --timeout-s 500 --warmup 0 --max-concurrency 8 \
      --output-dir $O/$name/video_pass$pass --disable-tqdm >> $O/$name.video.log 2>&1 &
    client=$!
    while kill -0 $client 2>/dev/null && kill -0 $driver 2>/dev/null; do sleep 2; done
    kill $client 2>/dev/null
    wait $client 2>/dev/null
    pass=$((pass + 1))
  done
}
limit_extra() { if [ -n "$1" ]; then echo "--preprocessing.factory.max_running_video_requests $1"; else echo ""; fi; }
closed() {
  local name=$1 tree=$2 card=$3 offset=$4 model=$5 videos=$6
  local cpus=$(cpus_of $card) port=$(port_of $card $offset) ballast=""
  mkdir -p $O/$name
  if [ $model = omni_colocated_h20_on_h200 ]; then
    python3 $K/ballast.py --card $card --free-mib $H20_FREE_MIB --ready-file $O/$name/ballast.txt > $O/$name/ballast.log 2>&1 &
    ballast=$!
    until [ -s $O/$name/ballast.txt ] || ! kill -0 $ballast 2>/dev/null; do sleep 2; done
    local extra="$(limit_extra $videos)"
  else
    local extra="$CONTEXT_32K $(limit_extra $videos)"
  fi
  note $name $tree "$model, limit ${videos:-default}, speech c16 closed loop beside 8 looping videos, card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
    --loads 16 --serve-extra "$extra" --omni-chunk-audio --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver $port $cpus &
  local looper=$!
  wait $driver
  local rc=$?
  wait $looper
  if [ -n "$ballast" ]; then kill $ballast 2>/dev/null; wait $ballast 2>/dev/null; else :; fi
  log "boot $name rc $rc videos $(python3 -c "import json,glob; print(sum(len(json.load(open(p)).get('results') or []) for p in glob.glob('$O/$name/video_pass*/videomme_results.json')))")"
}
until grep -q "q1128d_h done" $O/queue.log 2>/dev/null; do sleep 30; done
log "q1128d_i start"
H20=omni_colocated_h20_on_h200
both "closed hl1_m $M 0 6900 $H20 ''" "closed hl1_v2 $V 1 6900 $H20 2"
both "closed hl2_v2 $V 0 7100 $H20 2" "closed hl2_m $M 1 7100 $H20 ''"
both "closed hl3_m $M 0 7300 $H20 ''" "closed hl3_v4 $V 1 7300 $H20 4"
python3 $K/playback_margin.py $O/hl1_m $O/hl2_m $O/hl3_m $O/hl1_v2 $O/hl2_v2 $O/hl3_v4 >> $O/queue.log 2>&1
log "h20 limit read"
both "closed cs1_m $M 0 7500 omni_colocated ''" "closed cs1_v2 $V 1 7500 omni_colocated 2"
python3 $K/playback_margin.py $O/cs1_m $O/cs1_v2 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c16 "dev2 at limit 2 against main, speech c16 closed loop beside 8 looping videos, colocated H200" \
  main=cs1_m fix=cs1_v2 >> $O/queue.log 2>&1
log "h200 limit read"
log "q1128d_i done"
