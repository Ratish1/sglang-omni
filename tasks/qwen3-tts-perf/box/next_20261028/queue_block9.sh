#!/usr/bin/env bash
# Block 9 of 10-28: the two-frame minimum (A) on its new base before it opens. A's head acda990c9 (upstream main
# 1f0834e9c merged in) against that main: the qwen3_tts unit suite on A's head first, then the standard SeedTTS EN A/B
# (1.7B Base), one server per arm (serve_arm.py), same-time pairs then the cards swapped, every unseeded arm scored on
# its card. Cards 0 and 1: seeded c1 streaming (byte identity: A acts only where EOS would fall on frame 0 or 1), then
# non-streaming c16 in both orders. Cards 2 and 3: streaming c16 in both orders, then non-streaming c1.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
W=$D/a_newbase
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs
echo "queue block9 start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C /workspace/n28u_main rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')min2 $(git -C /workspace/n28u_min2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "== units min2: $(cd /workspace/n28u_min2 && CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider 2>&1 | tail -n 1) $(date +%T)" >> $L

arm() {
  local label=$1 tree=$2 card=$3 port=$4
  shift 4
  python3 $ARM --label $label --tree /workspace/n28u_$tree --card $card --port $port "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'RTF mean|WER corpus|similarity:|Throughput' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | tr '\n' ';') $(date +%T)" >> $L
}

pair() {
  local first=$1 second=$2
  arm $first &
  local a=$!
  arm $second &
  local b=$!
  wait $a $b
}

lane_one() {
  pair "c1s_main_seed main 0 41000 --concurrency 1 --seed 1234" "c1s_min2_seed min2 1 41500 --concurrency 1 --seed 1234"
  echo "== seeded c1 main against min2: $(python3 $ARM --compare c1s_main_seed c1s_min2_seed 2>&1 | tail -n 3 | tr '\n' ' ')" >> $L
  pair "c16n_main_a main 0 42000 --concurrency 16 --no-stream --score" "c16n_min2_a min2 1 42500 --concurrency 16 --no-stream --score"
  pair "c16n_min2_b min2 0 43000 --concurrency 16 --no-stream --score" "c16n_main_b main 1 43500 --concurrency 16 --no-stream --score"
  echo "lane one done $(date +%T)" >> $L
}

lane_two() {
  pair "c16s_main_c main 2 44000 --concurrency 16 --score" "c16s_min2_c min2 3 44500 --concurrency 16 --score"
  pair "c16s_min2_d min2 2 45000 --concurrency 16 --score" "c16s_main_d main 3 45500 --concurrency 16 --score"
  pair "c1n_main main 2 46000 --concurrency 1 --no-stream --score" "c1n_min2 min2 3 46500 --concurrency 1 --no-stream --score"
  echo "lane two done $(date +%T)" >> $L
}

lane_one &
one=$!
lane_two &
two=$!
wait $one $two
echo "queue block9 done $(date +%T)" >> $L
