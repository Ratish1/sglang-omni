#!/usr/bin/env bash
# The correctness stack against main on 1.7B Base SeedTTS EN, unseeded, each arm with its own tree's bench (main's sends
# no limit, the stack's its own 2048): non-streaming c16 as main, stack, stack, main; non-streaming c1 and streaming
# c16 as main, stack; every arm scored (WER, SIM) on the same card. Runs after queue_identity_3176.sh. Card 0 only.
set -u
D=${D:?the results directory}
AFTER=${AFTER:?the identity directory, for its done line}
MAIN=${MAIN:?the main tree}
STACK=${STACK:?the stack tree}
T=${T:?the kit tree}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
L=$D/queue.log
mkdir -p $D
until grep -q queue_identity_3176_done $AFTER/queue.log 2> /dev/null; do
  sleep 20
done
echo "queue stack ab start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')stack $(git -C $STACK rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
port=27000
for spec in "c16n_main1 $MAIN 16 --no-stream" "c16n_stack1 $STACK 16 --no-stream" "c16n_stack2 $STACK 16 --no-stream" \
  "c16n_main2 $MAIN 16 --no-stream" "c1n_main $MAIN 1 --no-stream" "c1n_stack $STACK 1 --no-stream" \
  "c16s_main $MAIN 16" "c16s_stack $STACK 16"; do
  set -- $spec
  ARM_ROOT=$D/arms python3 $ARM --label $1 --tree $2 --card 0 --port $port --concurrency $3 $4 --score >> $L 2>&1
  echo "arm $1 rc $? $(date +%T)" >> $L
  port=$((port + 20))
done
echo "queue_stack_ab_done $(date +%T)" >> $L
