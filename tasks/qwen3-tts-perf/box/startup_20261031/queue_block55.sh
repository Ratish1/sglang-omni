#!/usr/bin/env bash
# Block 55 of 10-31: two cold boots of the final startup stack, each into its own fresh cache, with nothing else on
# the box (block 54's cold boot overlapped a test run). Card 0. Waits for block 54 first.
# usage: D=... T=<kit tree> STACK=<tree> bash queue_block55.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
STACK=${STACK:?the stack tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block55
L=$W/queue.log
mkdir -p $W/logs
until grep -q "queue block54 done" $D/block54/queue.log 2> /dev/null; do sleep 20; done
echo "tree $STACK $(git -C $STACK rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $STACK diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
for round in 1 2; do
  cache=$W/cache_$round
  rm -rf $cache
  port=$((22000 + 200 * round))
  began=$(date +%s.%N)
  (cd $STACK && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$STACK TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_cold_$round.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port \
    --model-path $M17 --tts_engine.gpu 0 --vocoder.gpu 0" > $W/logs/serve_cold_$round.log 2>&1 < /dev/null &)
  sleep 1
  pgid=$(cat $W/logs/serve_cold_$round.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "cold_$round died" >> $L
      break
    else
      sleep 1
    fi
  done
  echo "cold_stack_$round healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
done
echo "queue block55 done $(date +%T)" >> $L
