#!/usr/bin/env bash
# 11-23, node-0001, after q1123_j. The full prefill graph alone on upstream main (24ef9b82c, the prototype cherry-picked
# onto 22bf41ef5, tree 1fe25fe4fd98), two processes, against main: does it help without one context? Decides whether it
# is its own PR landing first or part of part 2.
#   both cards: speech alone c16 and c64, same-time pairs swapped
#   card 0, card 1 idle: speech 8/s beside video 1/s, preprocessing max_concurrency 4, E M M E
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
UMAIN=/workspace/umain EM=/workspace/fullmain
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
FOUR="--preprocessing.factory.max_concurrency 4"
CPUS0=0-55,112-167 CPUS1=56-111,168-223
PORT0=30070
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
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "c16,c64 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16,64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
    --passes warm,events > $O/$name.driver.log 2>&1
  log "boot $name rc $? $(grep -m1 -o -E 'Traceback|Error[^ ]*' $O/$name/serve.log 2>/dev/null)"
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
mixed() {
  local name=$1 tree=$2
  log "host load $(cut -d' ' -f1-3 /proc/loadavg), $(node_free)"
  note $name $tree "seedtts 8/s seed $OMNI_ARRIVAL_SEED cap 64 beside video 1/s, preprocessing 4, card 0 port $PORT0"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $PORT0 --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K $FOUR" --omni-chunk-audio "--bench-extra=--request-rate=8" \
    --passes warm,events > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$PORT0/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  video_loop $name $tree $driver &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
  log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O $name)"
}
cd /workspace/sglang-omni
git fetch -q origin dev/qwen3-omni-full-prefill-on-main
git worktree add -q --detach $EM origin/dev/qwen3-omni-full-prefill-on-main 2>/dev/null \
  || git -C $EM checkout -q --detach origin/dev/qwen3-omni-full-prefill-on-main
log "tree fullmain $(git -C $EM rev-parse --short HEAD) $(git -C $EM rev-parse --short=12 HEAD:sglang_omni)"
until grep -q "q1123_j done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123_k start"
quiet_host
both "speech ka1_m0 $UMAIN 0 1610" "speech ka1_e1 $EM 1 1610"
quiet_host
both "speech ka2_e0 $EM 0 1630" "speech ka2_m1 $UMAIN 1 1630"
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "full prefill graph on main against main, $cell, same-time pairs swapped" \
    main=ka1_m0,ka2_m1 full=ka1_e1,ka2_e0 >> $O/queue.log 2>&1
done
python3 $K/first_audio_states.py --cell c16_events --main $O/ka1_m0,$O/ka2_m1 --branch $O/ka1_e1,$O/ka2_e0 \
  > $O/k_states_c16.txt 2>&1
grep -E "server first audio|thinker prefill|thinker chunks|frames until" $O/k_states_c16.txt >> $O/queue.log
log "K speech read"
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
mixed km1_e $EM
mixed km2_m $UMAIN
mixed km3_m $UMAIN
mixed km4_e $EM
python3 $K/body_table.py $O c64 "full prefill graph on main against main, beside video, preprocessing 4, card 0" \
  main=km2_m,km3_m full=km1_e,km4_e >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O km1_e km4_e km2_m km3_m >> $O/queue.log 2>&1
python3 $K/first_audio_states.py --cell c64_events --main $O/km2_m,$O/km3_m --branch $O/km1_e,$O/km4_e \
  > $O/k_states_mixed.txt 2>&1
grep -E "server first audio|preprocessing queue|thinker prefill|thinker chunks|frames until" $O/k_states_mixed.txt \
  >> $O/queue.log
log "K mixed read"
log "q1123_k done"
