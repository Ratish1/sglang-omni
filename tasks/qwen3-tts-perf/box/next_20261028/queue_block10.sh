#!/usr/bin/env bash
# Block 10 of 10-28: the reference encoder branch (D) on its new base before it opens. D's head 5bdc6f38d has the tree
# of f14c83813 (upstream main 1f0834e9c merged in), served from /workspace/n28u_refenc, against that main: the
# qwen3_tts unit suite, then the standard SeedTTS EN A/B (1.7B Base), one server per arm (serve_arm.py), same-time
# pairs then the cards swapped, every arm scored on its card. Cards 0 and 1: non-streaming c16 both orders, then
# non-streaming c1. Cards 2 and 3: streaming c16 both orders.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
W=$D/d_newbase
L=$W/queue.log
export ARM_ROOT=$W/arms
mkdir -p $ARM_ROOT $W/logs
echo "queue block10 start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C /workspace/n28u_main rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')refenc $(git -C /workspace/n28u_refenc rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
(cd /workspace/n28u_refenc && CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $W/units.txt 2>&1)
echo "== units refenc rc $?: $(grep -E '[0-9]+ passed|failed|error' $W/units.txt | tail -n 1) $(date +%T)" >> $L

arm() {
  local label=$1 tree=$2 card=$3 port=$4
  shift 4
  python3 $ARM --label $label --tree /workspace/n28u_$tree --card $card --port $port "$@" >> $W/logs/arm_$label.log 2>&1
  echo "arm $label rc $? $(grep -h -E 'RTF mean|WER \(corpus|Max token hits|Throughput' $ARM_ROOT/$label/bench.log $ARM_ROOT/$label/score*.log 2> /dev/null | tr -s ' ' | sort -u | tr '\n' ';') $(grep -h -o -E 'speaker similarity: [0-9.]+' $ARM_ROOT/$label/score*.log 2> /dev/null | tail -1) $(date +%T)" >> $L
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
  pair "c16n_main_a main 0 51000 --concurrency 16 --no-stream --score" "c16n_refenc_a refenc 1 51500 --concurrency 16 --no-stream --score"
  pair "c16n_refenc_b refenc 0 52000 --concurrency 16 --no-stream --score" "c16n_main_b main 1 52500 --concurrency 16 --no-stream --score"
  pair "c1n_main main 0 53000 --concurrency 1 --no-stream --score" "c1n_refenc refenc 1 53500 --concurrency 1 --no-stream --score"
  echo "lane one done $(date +%T)" >> $L
}

lane_two() {
  pair "c16s_main_c main 2 54000 --concurrency 16 --score" "c16s_refenc_c refenc 3 54500 --concurrency 16 --score"
  pair "c16s_refenc_d refenc 2 55000 --concurrency 16 --score" "c16s_main_d main 3 55500 --concurrency 16 --score"
  echo "lane two done $(date +%T)" >> $L
}

lane_one &
one=$!
lane_two &
two=$!
wait $one $two
echo "queue block10 done $(date +%T)" >> $L
