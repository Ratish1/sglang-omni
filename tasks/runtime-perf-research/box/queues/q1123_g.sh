#!/usr/bin/env bash
# 11-23, node-0001. E1 (5ee1c55f2: the thinker's prefill on the full graph, request slots = its 64 running requests,
# mixed chunk on) against F, the candidate from q1123_f's traced boots: thinker holds over 10 ms 792 to 73, the talker's
# turn wait 187 to 110 ms a second, every prefill replayed (1,748 replays, 0 eager). E2 (no mixed chunk) and E3 (its
# breakable control) lost the thinker's text cadence (first window median 163 and 178 ms against 150), so their pairs
# are dropped. The full graph changes FlashAttention's split rounding, so outputs move and the gates are quality.
#   card 1 now, beside q1123_e: seeded clean c16 and c64, E1 F F E1; then text Video-MME c1 and c16, two repeats, F, E1
#   both cards after q1123_e: same-time pairs E1 against F, c16 and c64, swapped
#   then card 1: Qwen3-ASR WER and SIM of the 8 seeded cells; card 0: E1 against F beside video, E1 F F E1, gated 649
set -u
O=/data/v1123
K=/workspace/kit/tasks/runtime-perf-research/box
F=/workspace/ocvg7 E1=/workspace/p-full-prefill-64
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS0=0-55,112-167 CPUS1=56-111,168-223
ASR_PORT=31470
PORT0=29270
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
seeded() {
  local name=$1 tree=$2
  note $name $tree "seeded clean c16,c64 card 1"
  taskset -c $CPUS1 python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port 31410 --out $O/$name \
    --loads 16,64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes clean "--bench-extra=--seed 3451" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
video_text() {
  local name=$1 tree=$2
  note $name $tree "omni_ci_onecard text Video-MME c1 c16 two repeats card 1"
  taskset -c $CPUS1 python3 $K/baseline.py --tree $tree --model omni_ci_onecard --card 1 --port 31430 --out $O/$name \
    --bench videomme_text --loads 1,16 --passes clean --bench-repeats 2 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
speech() {
  local name=$1 tree=$2 card=$3 offset=$4
  note $name $tree "c16,c64 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads 16,64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio \
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
  if [ -n "$p50" ] && [ "$p50" -le 649 ]; then
    log "$line, within 649"
  else
    log "$line, over 649: void, rerun as ${name}r, $(node_free)"
    mixed_once ${name}r $tree
    LAST=${name}r
    log "$(python3 $K/video_preprocess_gate.py --cell c64_events $O ${name}r)"
  fi
}
card1_first() {
  seeded gw1_e $E1
  seeded gw2_f $F
  seeded gw3_f $F
  seeded gw4_e $E1
  video_text gv1_f $F
  video_text gv2_e $E1
  for cell in c1 c16; do
    python3 $K/video_reps.py $O ${cell}_clean "text Video-MME $cell, F against E1" gv1_f gv2_e >> $O/queue.log 2>&1
  done
  python3 $K/videomme_text_identity.py --passes "F_c1=$O/gv1_f/c1_clean/bench/rep*/videomme_results.json" \
    "E1_c1=$O/gv2_e/c1_clean/bench/rep*/videomme_results.json" > $O/g_video_identity_c1.txt 2>&1
  tail -6 $O/g_video_identity_c1.txt >> $O/queue.log
  log "G video text read"
}
until grep -q "q1123_c done" $O/queue.log 2>/dev/null; do sleep 20; done
until ! pgrep -f "box/baseline.py .*--card 1 " > /dev/null; do sleep 20; done
log "q1123_g start"
card1_first &
c1=$!
until grep -q "q1123_e done" $O/queue.log 2>/dev/null; do sleep 20; done
wait $c1
quiet_host
both "speech ge1_f0 $F 0 1110" "speech ge1_e1 $E1 1 1110"
quiet_host
both "speech ge2_e0 $E1 0 1130" "speech ge2_f1 $F 1 1130"
for cell in c16 c64; do
  python3 $K/body_table.py $O $cell "E1 against F, $cell, same-time pairs swapped" F=ge1_f0,ge2_f1 E1=ge1_e1,ge2_e0 \
    >> $O/queue.log 2>&1
done
python3 $K/first_chunk_gap.py --cell c16_events $O F=ge1_f0,ge2_f1 E1=ge1_e1,ge2_e0 2>&1 | grep gaps >> $O/queue.log
python3 $K/first_audio_states.py --cell c16_events --main $O/ge1_f0,$O/ge2_f1 --branch $O/ge1_e1,$O/ge2_e0 \
  > $O/g_states_c16.txt 2>&1
grep -E "server first audio|thinker prefill|thinker chunks|frames until" $O/g_states_c16.txt >> $O/queue.log
log "G pairs read"
score() {
  cd $F
  taskset -c $CPUS1 setsid python3 -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 \
    --port $ASR_PORT --asr.gpu 1 > $O/asr_g_serve.log 2>&1 &
  echo $! > $O/asr_g.pgid
  until curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null || ! kill -0 $(cat $O/asr_g.pgid) 2>/dev/null; do
    sleep 5
  done
  for boot in gw1_e gw2_f gw3_f gw4_e; do
    for cell in c16 c64; do
      taskset -c $CPUS1 python3 -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --lang en --voice-clone --port $ASR_PORT \
        --asr-concurrency 64 --output-dir $O/$boot/${cell}_clean/bench > $O/$boot/${cell}_clean/transcribe.log 2>&1
      log "wer $boot $cell rc $?"
    done
  done
  kill -TERM -- -$(cat $O/asr_g.pgid) 2>/dev/null
  until [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; do sleep 5; done
  for boot in gw1_e gw2_f gw3_f gw4_e; do
    for cell in c16 c64; do
      taskset -c $CPUS1 python3 -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --lang en --voice-clone --device cuda:1 \
        --output-dir $O/$boot/${cell}_clean/bench > $O/$boot/${cell}_clean/similarity.log 2>&1
      log "similarity $boot $cell rc $? $(grep -m1 -E 'OutOfMemory' $O/$boot/${cell}_clean/similarity.log | cut -c1-60)"
    done
  done
  for cell in c16 c64; do
    python3 $K/wer_sim_cells.py E1_1=$O/gw1_e/${cell}_clean/bench F_2=$O/gw2_f/${cell}_clean/bench \
      F_3=$O/gw3_f/${cell}_clean/bench E1_4=$O/gw4_e/${cell}_clean/bench > $O/g_wer_$cell.txt 2>&1
    { echo "WER and SIM $cell"; cat $O/g_wer_$cell.txt; } >> $O/queue.log
  done
  log "G quality read"
}
score &
scorer=$!
mixed gm1_e $E1; m1=$LAST
mixed gm2_f $F; m2=$LAST
mixed gm3_f $F; m3=$LAST
mixed gm4_e $E1; m4=$LAST
python3 $K/body_table.py $O c64 "E1 against F, 8/s beside video, card 0 E1 F F E1" F=$m2,$m3 E1=$m1,$m4 >> $O/queue.log 2>&1
python3 $K/mixed_cell_load.py $O $m1 $m4 $m2 $m3 >> $O/queue.log 2>&1
python3 $K/first_audio_states.py --cell c64_events --main $O/$m2,$O/$m3 --branch $O/$m1,$O/$m4 > $O/g_states_mixed.txt 2>&1
grep -E "server first audio|preprocessing queue|thinker prefill|thinker chunks|frames until" $O/g_states_mixed.txt \
  >> $O/queue.log
log "G mixed read"
wait $scorer
log "q1123_g done"
