#!/usr/bin/env bash
# 11-23, node-0001. Replaces q1123_h and q1123_i (stopped at their first boot). Every beside-video read this lease
# (fix A on main, the three arms in q1123_e, E1 in q1123_g) moved first audio through the serial preprocessing queue's
# tail, a CPU stage none of the arms changes: E1's clean pair read every state from the audio encoder on level or
# better (thinker prefill -3.3 ms, first window -9.1 mean, -28.9 slowest 5 %) and the queue +66.9 ms mean with a lower
# median. As on 10-23, the discriminating run holds that stage fixed on both arms: preprocessing max_concurrency 4.
#   1. both cards: E1 (5ee1c55f2) against part 2's base BVG, same-time pairs swapped: c1, then c16 and c64
#   2. card 0, card 1 idle, preprocessing max_concurrency 4 on every boot: F BC NSP RIT E1 BVG BVG E1 RIT NSP BC F,
#      each arm against F's two boots, E1 and F against BVG. The 649 ms gate was set for serial preprocessing, so the
#      p50 is logged, not gated.
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/ocvg7 E1=/workspace/p-full-prefill-64 BVG=/workspace/bvg
BC=/workspace/x-blocking-copies NSP=/workspace/x-no-stream-priority RIT=/workspace/x-read-in-turn
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
FOUR="--preprocessing.factory.max_concurrency 4"
CPUS0=0-55,112-167 CPUS1=56-111,168-223
PORT0=29870
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
until ! pgrep -f "box/baseline.py" > /dev/null; do sleep 20; done
log "q1123_j start"
quiet_host
both "speech ja1_b0 $BVG 0 1410 1" "speech ja1_e1 $E1 1 1410 1"
quiet_host
both "speech ja2_e0 $E1 0 1430 1" "speech ja2_b1 $BVG 1 1430 1"
python3 $K/body_table.py $O c1 "E1 against part 2's base, c1, same-time pairs swapped" base=ja1_b0,ja2_b1 E1=ja1_e1,ja2_e0 \
  >> $O/queue.log 2>&1
quiet_host
both "speech jb1_b0 $BVG 0 1450 16,64" "speech jb1_e1 $E1 1 1450 16,64"
quiet_host
both "speech jb2_e0 $E1 0 1470 16,64" "speech jb2_b1 $BVG 1 1470 16,64"
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "E1 against part 2's base, $cell, same-time pairs swapped" base=jb1_b0,jb2_b1 \
    E1=jb1_e1,jb2_e0 >> $O/queue.log 2>&1
done
python3 $K/first_chunk_gap.py --cell c16_events $O base=jb1_b0,jb2_b1 E1=jb1_e1,jb2_e0 2>&1 | grep gaps >> $O/queue.log
python3 $K/first_audio_states.py --cell c16_events --main $O/jb1_b0,$O/jb2_b1 --branch $O/jb1_e1,$O/jb2_e0 \
  > $O/j_states_c16.txt 2>&1
grep -E "server first audio|thinker prefill|thinker chunks|frames until" $O/j_states_c16.txt >> $O/queue.log
log "J speech read"
mixed jm01_f $F
mixed jm02_bc $BC
mixed jm03_nsp $NSP
mixed jm04_rit $RIT
mixed jm05_e1 $E1
mixed jm06_b $BVG
mixed jm07_b $BVG
mixed jm08_e1 $E1
mixed jm09_rit $RIT
mixed jm10_nsp $NSP
mixed jm11_bc $BC
mixed jm12_f $F
for spec in "F blocking-copies jm01_f,jm12_f jm02_bc,jm11_bc" "F no-stream-priority jm01_f,jm12_f jm03_nsp,jm10_nsp" \
  "F read-in-turn jm01_f,jm12_f jm04_rit,jm09_rit" "F E1 jm01_f,jm12_f jm05_e1,jm08_e1" \
  "base E1 jm06_b,jm07_b jm05_e1,jm08_e1" "base F jm06_b,jm07_b jm01_f,jm12_f"; do
  set -- $spec
  python3 $K/body_table.py $O c64 "$2 against $1, 8/s beside video, preprocessing 4, card 0, card 1 idle" $1=$3 $2=$4 \
    >> $O/queue.log 2>&1
  python3 $K/first_audio_states.py --cell c64_events --main $O/${3%,*},$O/${3#*,} --branch $O/${4%,*},$O/${4#*,} \
    > $O/j_states_$1_$2.txt 2>&1
  grep -E "server first audio|preprocessing queue|thinker prefill|thinker chunks|frames until" $O/j_states_$1_$2.txt \
    >> $O/queue.log
done
python3 $K/mixed_cell_load.py $O jm01_f jm12_f jm02_bc jm11_bc jm03_nsp jm10_nsp jm04_rit jm09_rit jm05_e1 jm08_e1 \
  jm06_b jm07_b >> $O/queue.log 2>&1
log "J mixed read"
log "q1123_j done"
