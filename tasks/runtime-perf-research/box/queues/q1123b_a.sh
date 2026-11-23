#!/usr/bin/env bash
# 11-23 second lease, node-0001. The two reads lost when the first lease ended during q1123_j.
#   1. The full prefill graph alone on main (FM 24ef9b82c on upstream main 22bf41ef5, two processes) against main (M):
#      speech alone c16 and c64, same-time pairs on the two cards, swapped. Decides whether it is its own PR landing
#      first or part of part 2.
#   2. Card 0, card 1 idle, speech 8/s beside video 1/s, preprocessing max_concurrency 4 on every boot (the serial
#      preprocessing queue's tail decided every beside-video read with one stage, 11-23): symmetric order
#      F BC NSP RIT FM M M FM RIT NSP BC F. Fix A inside one context (BC), the read release (RIT) and the loop stream
#      priorities (NSP) against F's two boots, FM against M's two.
set -u
O=/data/v1123b
K=/workspace/kit/tasks/runtime-perf-research/box
M=/workspace/umain FM=/workspace/fullmain F=/workspace/ocvg7
BC=/workspace/x-blocking-copies NSP=/workspace/x-no-stream-priority RIT=/workspace/x-read-in-turn
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
FOUR="--preprocessing.factory.max_concurrency 4"
CPUS0=0-55,112-167 CPUS1=56-111,168-223
PORT0=28870
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
until grep -q "setup done" $O/queue.log 2>/dev/null; do sleep 20; done
log "q1123b_a start"
quiet_host
both "speech ka1_m0 $M 0 110" "speech ka1_f1 $FM 1 110"
quiet_host
both "speech ka2_f0 $FM 0 130" "speech ka2_m1 $M 1 130"
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "full prefill graph on main against main, $cell, same-time pairs swapped" \
    main=ka1_m0,ka2_m1 full=ka1_f1,ka2_f0 >> $O/queue.log 2>&1
done
python3 $K/first_audio_states.py --cell c16_events --main $O/ka1_m0,$O/ka2_m1 --branch $O/ka1_f1,$O/ka2_f0 \
  > $O/ka_states_c16.txt 2>&1
grep -E "server first audio|thinker prefill|thinker chunks|talker build|frames until" $O/ka_states_c16.txt >> $O/queue.log
python3 $K/first_chunk_gap.py --cell c16_events $O main=ka1_m0,ka2_m1 full=ka1_f1,ka2_f0 2>&1 | grep gaps >> $O/queue.log
log "KA read"
until ! pgrep -f "box/baseline.py" > /dev/null; do sleep 20; done
mixed km01_f $F
mixed km02_bc $BC
mixed km03_nsp $NSP
mixed km04_rit $RIT
mixed km05_fm $FM
mixed km06_m $M
mixed km07_m $M
mixed km08_fm $FM
mixed km09_rit $RIT
mixed km10_nsp $NSP
mixed km11_bc $BC
mixed km12_f $F
for spec in "F blocking-copies km01_f,km12_f km02_bc,km11_bc" "F no-stream-priority km01_f,km12_f km03_nsp,km10_nsp" \
  "F read-in-turn km01_f,km12_f km04_rit,km09_rit" "main full-on-main km06_m,km07_m km05_fm,km08_fm"; do
  set -- $spec
  python3 $K/body_table.py $O c64 "$2 against $1, 8/s beside video, preprocessing 4, card 0, card 1 idle" $1=$3 $2=$4 \
    >> $O/queue.log 2>&1
  python3 $K/first_audio_states.py --cell c64_events --main $O/${3%,*},$O/${3#*,} --branch $O/${4%,*},$O/${4#*,} \
    > $O/km_states_$2.txt 2>&1
  grep -E "server first audio|preprocessing queue|thinker prefill|thinker chunks|frames until" $O/km_states_$2.txt \
    >> $O/queue.log
done
python3 $K/mixed_cell_load.py $O km01_f km12_f km02_bc km11_bc km03_nsp km10_nsp km04_rit km09_rit km05_fm km08_fm \
  km06_m km07_m >> $O/queue.log 2>&1
log "KM read"
log "q1123b_a done"
