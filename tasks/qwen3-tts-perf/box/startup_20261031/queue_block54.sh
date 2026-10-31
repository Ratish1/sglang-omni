#!/usr/bin/env bash
# Block 54 of 10-31: (1) the Qwen3-TTS suite on DWARM (one warm-up per decoder and codec graph), card 1. (2) The final
# startup stack (A, B, C, M, E, DWARM) against main, card 0, one boot at a time: the stack cold into a fresh cache, then
# three rounds of warm boots of main (block 44's cache) and the stack, split by phase. Waits for block 53 first.
# usage: D=... T=<kit tree> MAIN=<tree> DWARM=<tree> STACK=<tree> bash queue_block54.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block54
L=$W/queue.log
mkdir -p $W/logs
until grep -q "queue block53 done" $D/block53/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN DWARM STACK; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block54 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L

(cd $DWARM && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$DWARM timeout 3600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/dwarm_suite.log 2>&1)
echo "dwarm_suite exit $? $(grep -E '(passed|failed|error)' $W/dwarm_suite.log | tail -n 1) $(date +%T)" >> $L

boot() {
  local label=$1 tree=$2 port=$3 cache=$4
  if (echo > /dev/tcp/127.0.0.1/$port) 2> /dev/null; then
    echo "$label port $port already answers, not started" >> $L
    return 1
  else
    :
  fi
  local began=$(date +%s.%N)
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port \
    --model-path $M17 --tts_engine.gpu 0 --vocoder.gpu 0" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 1
  local pgid=$(cat $W/logs/serve_$label.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "$label died; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 1
    fi
  done
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s, $(grep -o 'KV pool holds [0-9]* tokens' $W/logs/serve_$label.log | head -n 1) $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

rm -rf $W/cache_stack
boot cold_stack $STACK 21000 $W/cache_stack
port=21200
for round in 1 2 3; do
  boot warm_main_$round $MAIN $port $D/block44/boot_cache_main
  boot warm_stack_$round $STACK $((port + 200)) $W/cache_stack
  port=$((port + 400))
done
python3 $K/startup_20261031/boot_split.py $W > $W/split.txt 2>&1
tail -n 3 $W/split.txt >> $L
echo "queue block54 done $(date +%T)" >> $L
