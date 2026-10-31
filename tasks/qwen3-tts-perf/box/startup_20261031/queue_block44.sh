#!/usr/bin/env bash
# Block 44 of 10-31: PR A (HEAD: the codec's in-place Triton kernels as opaque ops in its compiled decode) against main
# (MAIN). (1) The touched suites on HEAD, card 1. (2) codec_forms.py on HEAD cold then warm in a fresh cache, and main
# warm on block 43's cache: the 12 compiled keys' replay time and waveform identity, one slot per row. (3) Default 1.7B
# Base boots on card 0, one at a time: main cold, HEAD cold, main warm, HEAD warm, each tree on its own fresh caches.
# (4) ab_runs.py, the standard SeedTTS A/B (non-streaming and streaming c16, both card orders).
# usage: D=... T=<kit tree> MAIN=<tree> HEAD=<tree> bash queue_block44.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
HEAD=${HEAD:?the PR tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block44
L=$W/queue.log
mkdir -p $W/logs
for tree in $MAIN $HEAD; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block44 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

(cd $HEAD && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$HEAD timeout 3600 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/qwen3_tts tests/unit_test/kernels > $W/suites.log 2>&1)
echo "suites exit $? $(tail -n 1 $W/suites.log) $(date +%T)" >> $L

forms() {
  local name=$1 tree=$2 cache=$3
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    timeout 1800 python3 $K/startup_20261031/codec_forms.py --out $W/$name > $W/logs/$name.log 2>&1)
  echo "$name exit $? $(date +%T)" >> $L
}

rm -rf $W/cache_head
forms head_cold $HEAD $W/cache_head
forms head_warm $HEAD $W/cache_head
forms main_warm $MAIN $D/block43/cache_main
python3 $K/startup_20261031/compare_forms.py $W main head > $W/compare.txt 2>&1

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
  python3 $K/startup_20261031/boot_phases.py $W/logs/serve_$label.log --min-gap 1.0 > $W/phases_$label.txt 2>&1
}

rm -rf $W/boot_cache_main $W/boot_cache_head
boot cold_main $MAIN 17000 $W/boot_cache_main
boot cold_head $HEAD 17200 $W/boot_cache_head
boot warm_main $MAIN 17400 $W/boot_cache_main
boot warm_head $HEAD 17600 $W/boot_cache_head

(cd $K/next_20261031 && python3 ab_runs.py --a main=$MAIN --b head=$HEAD --root $W/ab --port 11000 > $W/ab.txt 2>&1)
echo "ab: $(grep -E '^(n16|s16)_o[12]|paired' $W/ab.txt | cut -c1-400 | tr '\n' '#')" >> $L
echo "queue block44 done $(date +%T)" >> $L
