#!/usr/bin/env bash
# 11-15, node-0000: SeedTTS EN (voice clone) WER and SIM for #3451's final head (FINAL 17353ed22) against main
# 35890f10c, colocated. c64 is generated here as same-time pairs, swapped. c16 reuses two boots per arm already on this
# lease, made with the same client and flags: main p1_m0, p2_m1; the restructured code t1_t0, t2_t1 (7982b003a, whose
# later changes are names and the KV-full skip, a path those runs never reached). One Qwen3-ASR server on card 0
# transcribes every cell at 64 in flight from main's tree; speaker similarity runs on card 1 meanwhile.
set -u
O=/data/rp1115/quality
K=/workspace/kit2/tasks/runtime-perf-research/box
FINAL=/workspace/final MAIN=/workspace/main
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
CPUS0=0-55,112-167
CPUS1=56-111,168-223
ASR_PORT=21000
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
quiet_host() {
  until awk '{exit !($1 < 80)}' /proc/loadavg; do sleep 30; done
  log "host load $(cut -d' ' -f1-3 /proc/loadavg)"
}
generate() {
  local name=$1 tree=$2 card=$3
  local cpus=$CPUS0 port=28400
  if [ $card = 1 ]; then cpus=$CPUS1; port=30400; else :; fi
  note $name $tree "seedtts c64 clean card $card"
  taskset -c $cpus python3 $K/baseline.py --tree $tree --model omni_colocated --card $card --port $port --out $O/$name \
    --loads 64 --passes warm,clean --omni-chunk-audio > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
both() {
  eval "$1" &
  local a=$!
  eval "$2" &
  local b=$!
  wait $a $b
}
log "q1115_quality start"
quiet_host
both "generate g1_f0 $FINAL 0" "generate g1_m1 $MAIN 1"
quiet_host
both "generate g2_m0 $MAIN 0" "generate g2_f1 $FINAL 1"
CELLS="$O/g1_f0/c64_clean $O/g2_f1/c64_clean $O/g1_m1/c64_clean $O/g2_m0/c64_clean"
CELLS="$CELLS /data/rp1115/struct/t1_t0/c16_events /data/rp1115/struct/t2_t1/c16_events"
CELLS="$CELLS /data/rp1115/p1_m0/c16_events /data/rp1115/p2_m1/c16_events"
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
python3 $K/wer_sim_cells.py final_c64_g1=$O/g1_f0/c64_clean/bench final_c64_g2=$O/g2_f1/c64_clean/bench \
  main_c64_g1=$O/g1_m1/c64_clean/bench main_c64_g2=$O/g2_m0/c64_clean/bench \
  final_c16_t1=/data/rp1115/struct/t1_t0/c16_events/bench final_c16_t2=/data/rp1115/struct/t2_t1/c16_events/bench \
  main_c16_p1=/data/rp1115/p1_m0/c16_events/bench main_c16_p2=/data/rp1115/p2_m1/c16_events/bench >> $O/queue.log 2>&1
python3 $K/body_table.py $O c64 "final against main c64" main=g1_m1,g2_m0 final=g1_f0,g2_f1 >> $O/queue.log 2>&1
log "q1115_quality done"
