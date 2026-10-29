#!/usr/bin/env bash
# Block 15 of 10-29, after block 14 (WAIT_PID): the reference encoder speedups branch against main.
# perf/qwen3-tts-reference-encoder-compile 133b1e327 (main 08cecee99 plus integer conv padding and the encode compiled
# per graph bucket). Card 0: the qwen3_tts suite on the branch. Card 1: encoder_speedups_gate.py's served variant on the
# branch tree (its encode as served: device per encode, codes against main's and official's). Then the standard
# SeedTTS EN A/B (1.7B Base), same-time pairs on cards 0 and 1 then swapped, scored: non-streaming c16, streaming c16,
# non-streaming c1. Startup and the KV pool come from each arm's serve log.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
REPO=/workspace/sglang-omni
K=$T/tasks/qwen3-tts-perf/box/next_20261029
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block15
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
echo "queue block15 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
git -C $REPO fetch -q origin perf/qwen3-tts-reference-encoder-compile
rm -rf /workspace/n29_compile
git -C $REPO worktree prune
git -C $REPO worktree add -q --detach /workspace/n29_compile 133b1e327
echo "tree compile $(git -C /workspace/n29_compile rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L

(cd /workspace/n29_compile && CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $W/units_compile.txt 2>&1;
  echo "== units compile rc $?: $(grep -E '[0-9]+ passed|failed|error' $W/units_compile.txt | tail -n 1) $(date +%T)" >> $L) &
units=$!
(cd /workspace/n29_compile && PYTHONPATH=/workspace/n29_compile TORCHINDUCTOR_CACHE_DIR=$W/inductor_gate \
  python3 $K/encoder_speedups_gate.py $D/wavs --model $M17 --card 1 --variants served > $W/gate_compile.txt 2>&1;
  echo "gate compile rc $? $(grep -E '^== |official|base /' $W/gate_compile.txt | tr '\n' ';') $(date +%T)" >> $L) &
gate=$!
wait $units $gate

arm() {
  local label=$1 tree=$2 card=$3 port=$4
  shift 4
  python3 $ARM --label $label --tree /workspace/n29_$tree --card $card --port $port "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? startup $(cat $ARM_ROOT/$label/startup.txt 2> /dev/null) $(grep -h -E 'RTF mean|WER \(corpus|Max token hits|Throughput' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(date +%T)" >> $L
}

pair() {
  arm $1 &
  local a=$!
  arm $2 &
  local b=$!
  wait $a $b
}

pair "c16n_a_main main 0 11000 --concurrency 16 --no-stream --score" "c16n_a_compile compile 1 11500 --concurrency 16 --no-stream --score"
pair "c16n_b_compile compile 0 12000 --concurrency 16 --no-stream --score" "c16n_b_main main 1 12500 --concurrency 16 --no-stream --score"
pair "c16s_a_main main 0 13000 --concurrency 16 --score" "c16s_a_compile compile 1 13500 --concurrency 16 --score"
pair "c16s_b_compile compile 0 14000 --concurrency 16 --score" "c16s_b_main main 1 14500 --concurrency 16 --score"
pair "c1n_a_main main 0 17000 --concurrency 1 --no-stream --score" "c1n_a_compile compile 1 17500 --concurrency 1 --no-stream --score"
pair "c1n_b_compile compile 0 18000 --concurrency 1 --no-stream --score" "c1n_b_main main 1 18500 --concurrency 1 --no-stream --score"
for label in $(ls $ARM_ROOT); do
  echo "kv $label $(grep -h -o -E 'max_total_num_tokens=[0-9]+|#tokens: [0-9]+|KV Cache is allocated[^,]*' $ARM_ROOT/$label/serve.log | head -2 | tr '\n' ' ')" >> $L
done
echo "queue block15 done $(date +%T)" >> $L
