#!/usr/bin/env bash
# Block 40 of 10-31: two fixes, each tested where it acts.
# (1) The talker branch's spawned reader test (TALKER tree): it passes at the head, fails with the default-stream
#     hand-off replaced by a no-op, and the qwen3_tts suite passes.
# (2) The encode branch's tokenizer sharing (FIX tree against OLD, the encode PR's head before the fix): the qwen3_tts
#     suite; the engine test's reuse row fails on the loader before the fix (MUT: FIX with stages.py from the commit
#     before the loader change); then one-process 1.7B Base boots at the cascade profile (static 0.12), OLD on card 0
#     and FIX on card 1 at once, per layout: default; the vocoder given its own attention implementation (sdpa); the
#     vocoder told it does not share the engine's copy (OLD's and FIX's own kwarg names). Each boot logs the KV pool,
#     the tokenizer loads, reuses and float32 conversions, and its card's used memory at health.
# Ports below 32768, refused when one already answers.
# usage: D=... TALKER=<tree> FIX=<tree> MUT=<tree> OLD=<tree> bash queue_block40.sh
set -u
D=${D:?the results directory}
TALKER=${TALKER:?the talker tree}
FIX=${FIX:?the fixed encode tree}
MUT=${MUT:?the fixed encode tree with the loader before the fix}
OLD=${OLD:?the encode PR head before the fix}
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block40
L=$W/queue.log
READER=tests/unit_test/qwen3_tts/test_pipeline.py::test_qwen3_tts_another_process_reads_complete_final_codes
mkdir -p $W/logs
for tree in $TALKER $FIX $MUT $OLD; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block40 start $(date +%T)" >> $L

run_suite() {
  local label=$1 tree=$2 card=$3
  (cd $tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=$tree timeout 3600 python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/qwen3_tts > $W/logs/suite_$label.log 2>&1)
  echo "suite $label: $(tail -n 1 $W/logs/suite_$label.log) $(date +%T)" >> $L
}

run_suite talker $TALKER 0 &
a=$!
run_suite fix $FIX 1 &
b=$!
wait $a $b

(cd $TALKER && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TALKER python3 -m pytest -q -p no:cacheprovider --durations=1 $READER \
  > $W/logs/reader_head.log 2>&1)
echo "reader at the head: $(grep -E 'passed|failed' $W/logs/reader_head.log | tail -n 1); $(grep -E '[0-9.]+s call' $W/logs/reader_head.log | head -n 1)" >> $L
(cd $TALKER && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TALKER python3 - > $W/logs/reader_no_handoff.log 2>&1 <<PY
import sys

import pytest

from sglang_omni.models.qwen3_tts import request_builders

request_builders.hand_off_to_default_stream = lambda codes: None
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "$READER"]))
PY
)
echo "reader without the hand-off (must fail): $(grep -E 'passed|failed' $W/logs/reader_no_handoff.log | tail -n 1); $(grep -m 1 -E '^E +assert' $W/logs/reader_no_handoff.log)" >> $L

(cd $MUT && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$MUT python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/qwen3_tts/test_pipeline.py -k engine_attaches_its_process_speech_tokenizer > $W/logs/engine_rows_before_fix.log 2>&1)
echo "engine rows on the loader before the fix (the reuse row must fail): $(grep -E 'passed|failed' $W/logs/engine_rows_before_fix.log | tail -n 1); $(grep -o 'FAILED [^ ]*' $W/logs/engine_rows_before_fix.log | tr '\n' ' ')" >> $L

start_server() {
  local label=$1 tree=$2 port=$3 card=$4
  shift 4
  if (echo > /dev/tcp/127.0.0.1/$port) 2> /dev/null; then
    echo "$label port $port already answers, not started" >> $L
    return 1
  else
    :
  fi
  (cd $tree && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/logs/serve_$label.pid) waited=0
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label not healthy after $waited s; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  local log=$W/logs/serve_$label.log
  echo "$label healthy after $waited s: $(grep -o 'KV pool holds [0-9]* tokens' $log | head -n 1); loads $(grep -c 'Loading Qwen3-TTS speech tokenizer' $log), reuses $(grep -c 'Reusing the Qwen3-TTS speech tokenizer' $log), float32 on reuse $(grep -c 'Giving the reused Qwen3-TTS speech tokenizer' $log); card used $(nvidia-smi -i $card --query-gpu=memory.used --format=csv,noheader) $(date +%T)" >> $L
}

stop_server() {
  local pgid=$(cat $W/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

profile() {
  echo "--model-path $M17 --config examples/configs/qwen3_tts_1_7b.yaml --mem-fraction-static 0.12 --tts_engine.gpu $1 --vocoder.gpu $1"
}

for spec in "default 12000 12200||" \
  "attn 12400 12600|--vocoder.factory.attn_implementation sdpa|--vocoder.factory.attn_implementation sdpa" \
  "unshared 12800 13000|--vocoder.factory.reference_encoder_in_process false|--vocoder.factory.shares_engine_speech_tokenizer false"; do
  IFS='|' read -r head old_flags fix_flags <<< "$spec"
  set -- $head
  start_server ${1}_old $OLD $2 0 $(profile 0) $old_flags &
  a=$!
  start_server ${1}_fix $FIX $3 1 $(profile 1) $fix_flags &
  b=$!
  wait $a $b
  stop_server ${1}_old
  stop_server ${1}_fix
done
echo "queue block40 done $(date +%T)" >> $L
