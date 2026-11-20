#!/usr/bin/env bash
# 11-20, node-0000, both H200s: the talker's sampling state kept per request slot (SLOT d3806fb6a, tree 1a7c0b0636ac)
# and its stack with the preprocessing video lane (STACK 6149c714f, tree 0e5588dfb14c) against main (de4cb9f0f, tree
# 847f8a122c6f). Same-time pairs, one server per card on its own socket, arms swapped each round, a round starting when
# the host load is under 80. Seeded c1 identity first, in both card orders, then speech c16 and c64 with the talker
# probe (the step by row list decides whether a changed list still costs more), then SeedTTS c16 beside 8 looping videos
# on the colocated config (probed) and the CI layout, then WER and SIM of the speech cells.
set -u
O=/data/v1120/slot2
K=/workspace/kit/tasks/runtime-perf-research/box
MAIN=/workspace/tmain SLOT=/workspace/tslot STACK=/workspace/tstack
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
ASR_PORT=21000
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
cpus_of() { if [ $1 = 1 ]; then echo 56-111,168-223; else echo 0-55,112-167; fi; }
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
  local name=$1 tree=$2 card=$3
  note $name $tree "colocated seeded c1 card $card"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card 100) --out $O/$name --loads 1 --passes clean "--bench-extra=--seed 3451" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}

speech() {
  local name=$1 tree=$2 card=$3 load=$4 offset=$5
  note $name $tree "colocated c$load card $card probe"
  taskset -c $(cpus_of $card) python3 $K/baseline.py --tree $tree --model omni_colocated --card $card \
    --port $(port_of $card $offset) --out $O/$name --loads $load --passes warm,events --omni-chunk-audio --probe \
    --serve-env OMNI_HOP_PROBE_TALKER_BATCHES=1 > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}

mixed() {
  local name=$1 tree=$2 card=$3 model=$4 extra=$5 offset=$6 probe=${7:-}
  local cpus=$(cpus_of $card) port=$(port_of $card $offset)
  note $name $tree "$model seedtts c16 beside 8 videos card $card $probe"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model $model --card $card --port $port --out $O/$name \
    --loads 16 --passes warm,events --serve-extra "$extra" \
    ${probe:+--probe --serve-env OMNI_HOP_PROBE_TALKER_BATCHES=1} > $O/$name.driver.log 2>&1 &
  local driver=$!
  until curl -sf --max-time 2 http://127.0.0.1:$port/health > /dev/null || ! kill -0 $driver 2>/dev/null; do sleep 5; done
  (
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
  ) &
  local looper=$!
  wait $driver
  log "boot $name rc $?"
  wait $looper
}

log "q1120_slot start"
quiet_host
both "seeded id_s0 $SLOT 0" "seeded id_m1 $MAIN 1"
log "identity $(python3 $K/wav_identity.py $O c1_clean id_m1 id_s0 2>&1 | tail -1)"
quiet_host
both "seeded id2_m0 $MAIN 0" "seeded id2_s1 $SLOT 1"
log "identity $(python3 $K/wav_identity.py $O c1_clean id2_m0 id2_s1 2>&1 | tail -1)"

quiet_host
both "speech p1_s0 $SLOT 0 16 220" "speech p1_m1 $MAIN 1 16 220"
quiet_host
both "speech p2_m0 $MAIN 0 16 440" "speech p2_s1 $SLOT 1 16 440"
quiet_host
both "speech p3_s0 $SLOT 0 64 660" "speech p3_m1 $MAIN 1 64 660"
quiet_host
both "speech p4_m0 $MAIN 0 64 880" "speech p4_s1 $SLOT 1 64 880"
python3 $K/chunk_gap_parts.py --cell c16_events $O/p1_s0 $O/p2_s1 $O/p1_m1 $O/p2_m0 > $O/chunk_gap_parts_c16.txt 2>&1
python3 $K/chunk_gap_parts.py --cell c64_events $O/p3_s0 $O/p4_s1 $O/p3_m1 $O/p4_m0 > $O/chunk_gap_parts_c64.txt 2>&1
python3 $K/body_table.py $O c16 "slot against main c16" main=p1_m1,p2_m0 slot=p1_s0,p2_s1 >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "slot against main c64" main=p3_m1,p4_m0 slot=p3_s0,p4_s1 >> $O/queue.log 2>&1

quiet_host
both "mixed cv1_t0 $STACK 0 omni_colocated '$CONTEXT_32K' 1100 probe" \
  "mixed cv1_m1 $MAIN 1 omni_colocated '$CONTEXT_32K' 1100 probe"
quiet_host
both "mixed cv2_m0 $MAIN 0 omni_colocated '$CONTEXT_32K' 1320 probe" \
  "mixed cv2_t1 $STACK 1 omni_colocated '$CONTEXT_32K' 1320 probe"
quiet_host
both "mixed sv1_t0 $STACK 0 omni_ci_onecard '' 1540" "mixed sv1_m1 $MAIN 1 omni_ci_onecard '' 1540"
quiet_host
both "mixed sv2_m0 $MAIN 0 omni_ci_onecard '' 1760" "mixed sv2_t1 $STACK 1 omni_ci_onecard '' 1760"
python3 $K/chunk_gap_parts.py --cell c16_events $O/cv1_t0 $O/cv2_t1 $O/cv1_m1 $O/cv2_m0 > $O/chunk_gap_parts_cv.txt 2>&1
python3 $K/body_table.py $O c16 "stack against main, colocated beside video" main=cv1_m1,cv2_m0 stack=cv1_t0,cv2_t1 \
  >> $O/queue.log 2>&1
python3 $K/body_table.py $O c16 "stack against main, CI layout beside video" main=sv1_m1,sv2_m0 stack=sv1_t0,sv2_t1 \
  >> $O/queue.log 2>&1

CELLS="$O/p1_s0/c16_events $O/p2_s1/c16_events $O/p1_m1/c16_events $O/p2_m0/c16_events"
CELLS="$CELLS $O/p3_s0/c64_events $O/p4_s1/c64_events $O/p3_m1/c64_events $O/p4_m0/c64_events"
cd $MAIN
setsid python3 -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $ASR_PORT --asr.gpu 0 \
  > $O/asr_serve.log 2>&1 &
echo $! > $O/asr.pgid
until curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null || ! kill -0 $(cat $O/asr.pgid) 2>/dev/null; do
  sleep 5
done
log "asr server up $(curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null && echo yes || echo no)"
(
  for cell in $CELLS; do
    python3 -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --lang en --voice-clone --device cuda:1 \
      --output-dir $cell/bench > $cell/similarity.log 2>&1
    log "similarity $cell rc $?"
  done
) &
similarity_loop=$!
for cell in $CELLS; do
  python3 -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --lang en --voice-clone --port $ASR_PORT \
    --asr-concurrency 64 --output-dir $cell/bench > $cell/transcribe.log 2>&1
  log "wer $cell rc $?"
done
wait $similarity_loop
kill -TERM -- -$(cat $O/asr.pgid) 2>/dev/null
python3 $K/wer_sim_cells.py slot_c16_p1=$O/p1_s0/c16_events/bench slot_c16_p2=$O/p2_s1/c16_events/bench \
  main_c16_p1=$O/p1_m1/c16_events/bench main_c16_p2=$O/p2_m0/c16_events/bench \
  slot_c64_p3=$O/p3_s0/c64_events/bench slot_c64_p4=$O/p4_s1/c64_events/bench \
  main_c64_p3=$O/p3_m1/c64_events/bench main_c64_p4=$O/p4_m0/c64_events/bench >> $O/queue.log 2>&1
log "q1120_slot done"
