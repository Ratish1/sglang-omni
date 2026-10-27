#!/usr/bin/env bash
# The correctness stack against main on 1.7B Base SeedTTS EN, unseeded, each arm with its own tree's bench (main's sends
# no limit, the stack's its own 2048): non-streaming c16 as main, stack, stack, main; non-streaming c1 and streaming
# c16 as main, stack; every arm scored (WER, SIM) on the same card. ARMS keeps only the named arms. Waits for AFTER_LOG
# to hold AFTER_LINE. Card 0 only.
set -u
D=${D:?the results directory}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
MAIN=${MAIN:?the main tree}
STACK=${STACK:?the stack tree}
T=${T:?the kit tree}
ARMS=${ARMS:-"c16n_main1 c16n_stack1 c16n_stack2 c16n_main2 c1n_main c1n_stack c16s_main c16s_stack"}
PORT=${PORT:-27000}
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
L=$D/queue.log
mkdir -p $D
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
echo "queue stack ab start $(date +%T) arms [$ARMS] main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')stack $(git -C $STACK rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
port=$PORT
for spec in "c16n_main1 $MAIN 16 --no-stream" "c16n_stack1 $STACK 16 --no-stream" "c16n_stack2 $STACK 16 --no-stream" \
  "c16n_main2 $MAIN 16 --no-stream" "c1n_main $MAIN 1 --no-stream" "c1n_stack $STACK 1 --no-stream" \
  "c16s_main $MAIN 16" "c16s_stack $STACK 16" "c16s_stack2 $STACK 16" "c16s_main2 $MAIN 16" \
  "c64s_main $MAIN 64" "c64s_stack $STACK 64" "c64s_stack2 $STACK 64" "c64s_main2 $MAIN 64"; do
  set -- $spec
  if [[ " $ARMS " != *" $1 "* ]]; then
    port=$((port + 20))
    continue
  else
    :
  fi
  if [ -f $D/arms/$1/bench/similarity_results.json ]; then
    echo "arm $1 already scored, skipped $(date +%T)" >> $L
  else
    ARM_ROOT=$D/arms python3 $ARM --label $1 --tree $2 --card 0 --port $port --concurrency $3 ${4:-} --score >> $L 2>&1
    echo "arm $1 rc $? $(date +%T)" >> $L
  fi
  port=$((port + 20))
done
echo "queue_stack_ab_done $(date +%T)" >> $L
