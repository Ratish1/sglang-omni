#!/usr/bin/env bash
# Block 45 of 10-31: PR B (HEAD: the codec decode compiled once per size class, batch and width symbolic) against its
# base PR A (BASE). (1) PR B's new test on HEAD and on BASE (where it must fail), then the touched suites on HEAD, card 1.
# (2) codec_forms.py cold then warm on HEAD and on MOD (HEAD with main's modulo for the dilation tail), each in a fresh
# cache, and BASE warm on block 44's cache: programs, compile seconds, replay time, waveforms against BASE. (3) Default
# 1.7B Base boots on card 0, one at a time: BASE cold, HEAD cold, BASE warm, HEAD warm, each tree on fresh caches.
# (4) ab_runs.py, the standard SeedTTS A/B (non-streaming and streaming c16, both card orders), BASE against HEAD.
# Waits for block 44's queue to finish first.
# usage: D=... T=<kit tree> BASE=<tree> HEAD=<tree> MOD=<tree> bash queue_block45.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the PR A tree}
HEAD=${HEAD:?the PR B tree}
MOD=${MOD:?the modulo tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
TEST=tests/unit_test/qwen3_tts/test_incremental_codec.py::test_compiled_decode_serves_every_batch_and_width_above_one_with_one_program
W=$D/block45
L=$W/queue.log
mkdir -p $W/logs
until grep -q "queue block44 done" $D/block44/queue.log 2> /dev/null; do sleep 20; done
for tree in $BASE $HEAD $MOD; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block45 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

(cd $HEAD && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$HEAD python3 -m pytest -q -p no:cacheprovider $TEST > $W/test_head.log 2>&1)
echo "new test on HEAD exit $? $(grep -E 'passed|failed' $W/test_head.log | tail -n 1)" >> $L
cp $HEAD/tests/unit_test/qwen3_tts/test_incremental_codec.py $BASE/tests/unit_test/qwen3_tts/test_incremental_codec.py
(cd $BASE && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$BASE python3 -m pytest -q -p no:cacheprovider $TEST > $W/test_base.log 2>&1)
echo "new test on BASE exit $? $(grep -E 'passed|failed' $W/test_base.log | tail -n 1) $(grep -E '^E ' $W/test_base.log | head -n 2 | tr '\n' ' ')" >> $L
git -C $BASE checkout -q -- tests
(cd $HEAD && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$HEAD timeout 3600 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/qwen3_tts tests/unit_test/kernels > $W/suites.log 2>&1)
echo "suites exit $? $(grep -E 'passed|failed' $W/suites.log | tail -n 1) $(date +%T)" >> $L

forms() {
  local name=$1 tree=$2 cache=$3
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    timeout 1800 python3 $K/startup_20261031/codec_forms.py --out $W/$name > $W/logs/$name.log 2>&1)
  echo "$name exit $? $(date +%T)" >> $L
}

rm -rf $W/cache_head $W/cache_mod
forms head_cold $HEAD $W/cache_head
forms head_warm $HEAD $W/cache_head
forms mod_cold $MOD $W/cache_mod
forms mod_warm $MOD $W/cache_mod
forms base_warm $BASE $D/block44/cache_head
python3 $K/startup_20261031/compare_forms.py $W base head mod > $W/compare.txt 2>&1

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

rm -rf $W/boot_cache_base $W/boot_cache_head
boot cold_base $BASE 17800 $W/boot_cache_base
boot cold_head $HEAD 18000 $W/boot_cache_head
boot warm_base $BASE 18200 $W/boot_cache_base
boot warm_head $HEAD 18400 $W/boot_cache_head

(cd $K/next_20261031 && python3 ab_runs.py --a base=$BASE --b head=$HEAD --root $W/ab --port 11400 > $W/ab.txt 2>&1)
echo "ab: $(grep -E '^(n16|s16)_o[12]|paired' $W/ab.txt | cut -c1-400 | tr '\n' '#')" >> $L
echo "queue block45 done $(date +%T)" >> $L
