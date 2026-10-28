#!/usr/bin/env bash
# Block 2 of 10-28: the standard SeedTTS EN A/B (1.7B Base) of the encoder branch and the two-frame minimum, each
# against main on its own base, same-time pairs on two cards and then the cards swapped, one server per arm
# (serve_arm.py), every unseeded arm scored (WER, SIM) on its card.
# Cards 0 and 1, the encoder branch d1d7dc79e: non-streaming c16 and streaming c16 in both orders, non-streaming c1.
# Cards 2 and 3, the two-frame minimum 221b71346: seeded c1 streaming (byte identity: the change acts only where EOS
# would fall on frame 0 or 1), streaming c16 in both orders. Waits for block 1b.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
L=$D/queue_2.log
export ARM_ROOT=$D/arms
mkdir -p $ARM_ROOT
until grep -q "queue block1b done" $D/queue_b.log 2> /dev/null; do sleep 20; done
echo "queue block2 start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C /workspace/n28_main rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')refenc2 $(git -C /workspace/n28_refenc2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')min2 $(git -C /workspace/n28_min2 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L

arm() {
  local label=$1 tree=$2 card=$3 port=$4
  shift 4
  python3 $ARM --label $label --tree /workspace/n28_$tree --card $card --port $port "$@" >> $D/logs/arm_$label.log 2>&1
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

refenc_lane() {
  pair "c16n_main_a main 0 21000 --concurrency 16 --no-stream --score" "c16n_refenc_a refenc2 1 21500 --concurrency 16 --no-stream --score"
  pair "c16n_refenc_b refenc2 0 22000 --concurrency 16 --no-stream --score" "c16n_main_b main 1 22500 --concurrency 16 --no-stream --score"
  pair "c16s_main_a main 0 23000 --concurrency 16 --score" "c16s_refenc_a refenc2 1 23500 --concurrency 16 --score"
  pair "c16s_refenc_b refenc2 0 24000 --concurrency 16 --score" "c16s_main_b main 1 24500 --concurrency 16 --score"
  pair "c1n_main main 0 25000 --concurrency 1 --no-stream --score" "c1n_refenc refenc2 1 25500 --concurrency 1 --no-stream --score"
  echo "refenc lane done $(date +%T)" >> $L
}

min2_lane() {
  pair "c1s_main_seed main 2 26000 --concurrency 1 --seed 1234" "c1s_min2_seed min2 3 26500 --concurrency 1 --seed 1234"
  echo "== seeded c1 main against min2: $(python3 $ARM --compare c1s_main_seed c1s_min2_seed 2>&1 | tail -n 3 | tr '\n' ' ')" >> $L
  pair "c16s_main_c main 2 27000 --concurrency 16 --score" "c16s_min2_c min2 3 27500 --concurrency 16 --score"
  pair "c16s_min2_d min2 2 28000 --concurrency 16 --score" "c16s_main_d main 3 28500 --concurrency 16 --score"
  echo "min2 lane done $(date +%T)" >> $L
}

refenc_lane &
r=$!
min2_lane &
m=$!
wait $r $m
echo "queue block2 done $(date +%T)" >> $L
