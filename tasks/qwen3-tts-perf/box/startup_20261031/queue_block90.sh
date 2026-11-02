#!/usr/bin/env bash
# Block 90 of 11-02: the last reads for whole requests on the codec state, against #3418's final head, cards 0 and 1 of
# the new lease. The touched test file; the standard SeedTTS A/B, scored, both card orders; first chunks with streaming
# and non-streaming clients on one server, both card orders; 1.7B CustomVoice non-streaming, seeded at concurrency 1 on
# all prompts, scored, the no-reference path no earlier block served. The speed at c64 and the seeded Base quality are
# block 89's and are not repeated.
# usage: D=... T=<kit tree> PR=<tree> WHOLE=<tree> bash queue_block90.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
PR=${PR:?the base tree}
WHOLE=${WHOLE:?the tree under test}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block90
L=$W/queue.log
mkdir -p $W
for arm in PR WHOLE; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block90 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $WHOLE && CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 1800 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 12 | cut -c1-160 >> $L

(cd $K/next_20261031 && python3 ab_runs.py --a pr=$PR --b whole=$WHOLE --root $W/ab --port 21000 > $W/ab.txt 2>&1)
echo "ab rc $? $(date +%T)" >> $L
grep -E '^(n16|s16)_o[12]|paired|Error' $W/ab.txt | cut -c1-300 >> $L

(cd $K/startup_20261031 && python3 mixed_load.py --a pr=$PR --b whole=$WHOLE --root $W/mixed --port 22000 > $W/mixed.txt 2>&1)
echo "mixed rc $? $(date +%T)" >> $L
grep -E 'first chunk|Error' $W/mixed.txt | cut -c1-300 >> $L

run_voice() {
  local label=$1 tree=$2 card=$3 port=$4
  (cd $K/inline_20261026 && ARM_ROOT=$W/voice TORCHINDUCTOR_CACHE_DIR=$W/cache_voice_$label/inductor TRITON_CACHE_DIR=$W/cache_voice_$label/triton timeout 3000 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency 1 --seed 1234 --no-stream --score --barrier $W/voice_barrier --parties 2 \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice "--bench-args=--task-type CustomVoice --no-ref-audio" > $W/voice_$label.out 2>&1)
  echo "voice $label card $card exit $? $(date +%T) startup $(cat $W/voice/$label/startup.txt 2>/dev/null) s; $(grep -hE 'Completed requests|Failed requests|RTF mean|Latency mean|WER \(corpus' $W/voice/$label/bench.log $W/voice/$label/score--transcribe-only.log 2>/dev/null | tr -s ' ' | tr '\n' ';' | cut -c1-220); tracebacks $(grep -c Traceback $W/voice/$label/serve.log)" >> $L
}
run_voice pr $PR 0 23000 &
run_voice whole $WHOLE 1 23800 &
wait
echo "queue block90 done $(date +%T)" >> $L
