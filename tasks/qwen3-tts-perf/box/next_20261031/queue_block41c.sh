#!/usr/bin/env bash
# Block 41c of 10-31: block 41's boots and A/B on the compile PR's head with Mimi's ceil-division padding (HEAD) against
# main (MAIN). Boots, MAIN card 0 and HEAD card 1 at once, each tree on its own fresh inductor and Triton caches: default
# 1.7B Base cold then warm, 1.7B CustomVoice, the cascade profiles (0.6B static 0.08, 1.7B static 0.12): health time, KV
# pool, encoder graph lines. Then ab_runs.py, the standard SeedTTS A/B (non-streaming and streaming c16, both orders).
# Ports 17000 to 19900 for boots and 11000 for the A/B, refused when one already answers.
# usage: D=... T=<kit tree> MAIN=<tree> HEAD=<tree> bash queue_block41c.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
HEAD=${HEAD:?the compile head tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block41c
L=$W/queue.log
mkdir -p $W/logs
for tree in $MAIN $HEAD; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block41c start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

boot() {
  local label=$1 tree=$2 port=$3 cache=$4
  shift 4
  if (echo > /dev/tcp/127.0.0.1/$port) 2> /dev/null; then
    echo "$label port $port already answers, not started" >> $L
    return 1
  else
    :
  fi
  local began=$(date +%s.%N)
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" \
    > $W/logs/serve_$label.log 2>&1 < /dev/null &)
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
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s, $(grep -o 'KV pool holds [0-9]* tokens' $W/logs/serve_$label.log | head -n 1), encoder graph lines $(grep -c 'reference encoder graphs captured' $W/logs/serve_$label.log) $(date +%T)" >> $L
}

stop() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

pair() {
  local name=$1 port=$2
  shift 2
  boot ${name}_main $MAIN $port $W/boot_cache_main --tts_engine.gpu 0 --vocoder.gpu 0 $* &
  local a=$!
  boot ${name}_head $HEAD $((port + 200)) $W/boot_cache_head --tts_engine.gpu 1 --vocoder.gpu 1 $* &
  local b=$!
  wait $a $b
  stop ${name}_main
  stop ${name}_head
}

rm -rf $W/boot_cache_main $W/boot_cache_head
pair cold 17000 --model-path $M17
pair warm 17600 --model-path $M17
pair customvoice 18200 --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
pair cascade06 18800 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --config examples/configs/qwen3_tts_0_6b.yaml --mem-fraction-static 0.08
pair cascade17 19400 --model-path $M17 --config examples/configs/qwen3_tts_1_7b.yaml --mem-fraction-static 0.12

(cd $K/next_20261031 && python3 ab_runs.py --a main=$MAIN --b head=$HEAD --root $W/ab --port 11000 > $W/ab.txt 2>&1)
echo "ab: $(grep -E '^(n16|s16)_o[12]|paired' $W/ab.txt | cut -c1-400 | tr '\n' '#')" >> $L
echo "queue block41c done $(date +%T)" >> $L
