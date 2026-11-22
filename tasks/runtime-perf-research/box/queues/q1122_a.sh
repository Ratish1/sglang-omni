#!/usr/bin/env bash
# 11-22, node-0001, two H200. Two open items of part 2 (DESIGN_one_context_speech.md section 7):
#   T (card 1)  part 2's c16 first audio p99 (+13.0 % on its base): speech alone c16 with the talker batch, thinker step
#               and turn probes, one context (OCVG 7cae415a1) then two processes (BVG 6aff09f46); per request, what
#               fills the frames until code2wav's first window, the state the slowest 5 % lose 38 ms in.
#   W (card 0)  the combined form's c64 WER below 50 % (1.84 against main's 1.66, unseeded): seeded c64 speech alone for
#               main (NMAIN e0482a08c), part 2's base (BVG) and combined (OCVG), two boots each, so sampling draws are
#               the same per sample in every arm and only numerics move; wav identity within and across arms.
#   S (card 1)  WER and SIM of W's six cells, Qwen3-ASR on card 1.
set -u
O=/data/v1122
K=/workspace/kit/tasks/runtime-perf-research/box
NMAIN=/workspace/nmain BVG=/workspace/bvg OCVG=/workspace/ocvg
CONTEXT_32K="--preprocessing.factory.max_seq_len 32768 --thinker.factory.max_seq_len 32768"
CPUS0=0-55,112-167 CPUS1=56-111,168-223
ASR_PORT=30500
TRACE_PROBES="OMNI_HOP_PROBE_TALKER_BATCHES=1 OMNI_HOP_PROBE_THINKER_STEPS=1 OMNI_HOP_PROBE_TURNS=1"
export SGLANG_OMNI_STARTUP_TIMEOUT=3600
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
note() { log "boot $1 $(git -C $2 rev-parse --short HEAD) $(git -C $2 rev-parse --short=12 HEAD:sglang_omni) $3 start"; }
traced() {
  local name=$1 tree=$2 port=$3
  note $name $tree "seedtts c16 alone card 1 traced port $port"
  taskset -c $CPUS1 python3 $K/baseline.py --tree $tree --model omni_colocated --card 1 --port $port --out $O/$name \
    --loads 16 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes warm,events --probe --serve-env "$TRACE_PROBES" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
seeded64() {
  local name=$1 tree=$2 port=$3
  note $name $tree "seeded c64 alone card 0 port $port"
  taskset -c $CPUS0 python3 $K/baseline.py --tree $tree --model omni_colocated --card 0 --port $port --out $O/$name \
    --loads 64 --serve-extra "$CONTEXT_32K" --omni-chunk-audio --passes clean "--bench-extra=--seed 3451" \
    > $O/$name.driver.log 2>&1
  log "boot $name rc $?"
}
until grep -q "^setup done" $O/queue.log 2>/dev/null; do sleep 20; done
for tree in $NMAIN $BVG $OCVG; do
  if ! git -C $tree diff --quiet HEAD -- sglang_omni; then
    log "q1122_a stopped: $tree is dirty"
    exit 1
  else :; fi
done
log "q1122_a start"
(
  traced tb1 $OCVG 31490
  traced ta1 $BVG 31450
  for boot in tb1 ta1; do
    python3 $K/tail_first_window.py $O/$boot --cell c16_events > $O/$boot.first_window.txt 2>&1
  done
  python3 $K/first_audio_states.py --cell c16_events --main $O/ta1 --branch $O/tb1 > $O/t_states_c16.txt 2>&1
  python3 $K/turn_waits.py $O/tb1 --cell c16_events > $O/tb1.turns_c16.txt 2>&1
  python3 $K/turn_hold_parts.py $O/tb1 --cell c16_events > $O/tb1.holds_c16.txt 2>&1
  grep -h "slowest\|all (median)" $O/tb1.first_window.txt $O/ta1.first_window.txt >> $O/queue.log
  log "T read"
) &
trace_section=$!
seeded64 ws1_b $OCVG 28290
seeded64 ws2_a $NMAIN 28290
seeded64 ws3_v $BVG 28290
seeded64 ws4_v $BVG 28290
seeded64 ws5_a $NMAIN 28290
seeded64 ws6_b $OCVG 28290
for pair in "ws2_a ws5_a" "ws3_v ws4_v" "ws1_b ws6_b" "ws2_a ws3_v" "ws3_v ws1_b" "ws2_a ws1_b"; do
  set -- $pair
  log "identity $1 $2 $(python3 $K/wav_identity.py $O c64_clean $1 $2 2>&1 | tail -1)"
done
log "W boots done"
wait $trace_section
cd $NMAIN
setsid python3 -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $ASR_PORT --asr.gpu 1 \
  > $O/asr_serve.log 2>&1 &
echo $! > $O/asr.pgid
until curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null || ! kill -0 $(cat $O/asr.pgid) 2>/dev/null; do
  sleep 5
done
log "asr server up $(curl -sf --max-time 2 http://127.0.0.1:$ASR_PORT/health > /dev/null && echo yes || echo no)"
for boot in ws1_b ws2_a ws3_v ws4_v ws5_a ws6_b; do
  cell=$O/$boot/c64_clean
  python3 -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --lang en --voice-clone --port $ASR_PORT \
    --asr-concurrency 64 --output-dir $cell/bench > $cell/transcribe.log 2>&1
  log "wer $boot rc $?"
done
kill -TERM -- -$(cat $O/asr.pgid) 2>/dev/null
for boot in ws1_b ws2_a ws3_v ws4_v ws5_a ws6_b; do
  cell=$O/$boot/c64_clean
  python3 -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --lang en --voice-clone --device cuda:1 \
    --output-dir $cell/bench > $cell/similarity.log 2>&1
  log "similarity $boot rc $? $(grep -m1 -E 'Error|error' $cell/similarity.log | cut -c1-120)"
done
python3 $K/wer_sim_cells.py comb_1=$O/ws1_b/c64_clean/bench main_2=$O/ws2_a/c64_clean/bench base_3=$O/ws3_v/c64_clean/bench \
  base_4=$O/ws4_v/c64_clean/bench main_5=$O/ws5_a/c64_clean/bench comb_6=$O/ws6_b/c64_clean/bench >> $O/queue.log 2>&1
log "S read"
log "q1122_a done"
