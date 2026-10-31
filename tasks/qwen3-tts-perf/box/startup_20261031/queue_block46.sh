#!/usr/bin/env bash
# Block 46 of 10-31: the CPU-side startup fixes, each alone on main and stacked. Arms: main; LAUNCHER (the launcher
# reads Qwen3-TTS's admission defaults without importing the engine); MING (the Ming talker package stops importing
# its audio VAE, so registering the omni models skips diffusers); FREEZE (the stage process freezes its heap before
# constructing each stage); STACK (PR B plus the three). Default 1.7B Base, card 0, one boot at a time. The first four
# compile the same codec as main, so they boot warm on block 44's main cache; STACK boots cold once into its own cache
# first. Then three rounds of warm boots over the five arms, each with the log timeline. Waits for block 45 first.
# usage: D=... T=<kit tree> MAIN=<tree> LAUNCHER=<tree> MING=<tree> FREEZE=<tree> STACK=<tree> bash queue_block46.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block46
L=$W/queue.log
mkdir -p $W/logs
until grep -q "queue block45 done" $D/block45/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN LAUNCHER MING FREEZE STACK; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block46 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

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
  python3 $K/startup_20261031/boot_phases.py $W/logs/serve_$label.log --min-gap 0.4 > $W/phases_$label.txt 2>&1
}

rm -rf $W/cache_stack
boot cold_stack $STACK 19000 $W/cache_stack
port=19200
for round in 1 2 3; do
  for arm in MAIN LAUNCHER MING FREEZE STACK; do
    cache=$D/block44/boot_cache_main
    if [ $arm = STACK ]; then
      cache=$W/cache_stack
    else
      :
    fi
    boot warm_${arm,,}_$round ${!arm} $port $cache
    port=$((port + 200))
  done
done
echo "queue block46 done $(date +%T)" >> $L
