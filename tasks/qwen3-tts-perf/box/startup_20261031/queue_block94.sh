#!/usr/bin/env bash
# Block 94 of 11-02: whole requests on the codec state for 1.7B CustomVoice and VoiceDesign, against #3418's head, which
# decode through the eager tokenizer path. The touched test file. Same-time pairs, one server per card, all SeedTTS EN
# prompts: CustomVoice non-streaming at concurrency 64 in both card orders and 16 once, seeded concurrency 1 scored for WER;
# VoiceDesign non-streaming at concurrency 16, scored. Base's numbers are blocks 88 to 92 and are not repeated.
# usage: D=... T=<kit tree> PR=<tree> WHOLE=<tree> bash queue_block94.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
PR=${PR:?the base tree}
WHOLE=${WHOLE:?the tree under test}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block94
L=$W/queue.log
mkdir -p $W
for arm in PR WHOLE; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block94 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $WHOLE && CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 1800 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 12 | cut -c1-160 >> $L

run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 concurrency=$5 barrier=$6 model=$7 bench_args=$8 flags=$9
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 3000 python3 serve_arm.py --label $label --tree $tree --card $card --port $port --concurrency $concurrency --no-stream $flags \
    --barrier $barrier --parties 2 --model $model "--bench-args=$bench_args" > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null) s; $(python3 - $W/arms/$label/bench/speed_results.json <<'EOF'
import json
import sys
try:
    s = json.load(open(sys.argv[1]))["summary"]
    print(f"done {s['completed_requests']}/{s['total_requests']} req/s {s['throughput_qps']:.2f} RTF {s['rtf_mean']:.4f} latency mean {1000 * s['latency_mean_s']:.1f} p95 {1000 * s['latency_p95_s']:.1f} ms")
except Exception as exc:
    print("no summary", exc)
EOF
); $(grep -hE 'WER \(corpus' $W/arms/$label/score--transcribe-only.log 2>/dev/null | tr -s ' ' | cut -c1-60); $(grep -m1 -oE '[0-9]+ slots' $W/arms/$label/serve.log); $(grep -m1 -oE '#tokens: [0-9]+' $W/arms/$label/serve.log); tracebacks $(grep -c 'Traceback' $W/arms/$label/serve.log); left context $(grep -c 'left-context decoder' $W/arms/$label/serve.log)" >> $L
}

VOICE=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
DESIGN=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
VOICE_ARGS="--task-type CustomVoice --no-ref-audio"
DESIGN_ARGS="--task-type VoiceDesign --no-ref-audio --instructions=A_calm_adult_female_voice_speaking_clearly"
port=20000
pair() {
  local name=$1 order=$2 concurrency=$3 model=$4 bench_args=$5 flags=$6
  run_arm ${name}_pr $PR $order $port $concurrency $W/barrier_$name $model "$bench_args" "$flags" &
  run_arm ${name}_whole $WHOLE $((1 - order)) $((port + 200)) $concurrency $W/barrier_$name $model "$bench_args" "$flags" &
  wait
  port=$((port + 1000))
}
pair voice_n64_o0 0 64 $VOICE "$VOICE_ARGS" ""
pair voice_n64_o1 1 64 $VOICE "$VOICE_ARGS" ""
pair voice_n16_o0 0 16 $VOICE "$VOICE_ARGS" ""
pair voice_c1_seeded 0 1 $VOICE "$VOICE_ARGS" "--seed 1234 --score"
pair design_n16_o0 0 16 $DESIGN "$DESIGN_ARGS" "--score"
echo "queue block94 done $(date +%T)" >> $L
