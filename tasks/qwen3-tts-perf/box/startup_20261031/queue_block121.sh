#!/usr/bin/env bash
# Block 121 of 11-04: a whole request one window replay at a time, against the 256-frame tree. The touched suite. Seeded c1 in
# both modes: a row alone replays the same windows, so anything but byte identical stops it. First chunks beside long requests,
# both orders, against block 120b's band. SeedTTS non-streaming c64 and c16, both orders.
# usage: D=... T=<kit tree> BASE=<tree> ARM=<tree> HARNESS=<tree> [BLOCK=block121] [PORT=12000] bash queue_block121.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the 256 tree}
ARM=${ARM:?the tree under test}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block121}
PORT=${PORT:-12000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in BASE ARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $ARM && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
(cd $K/next_20261031 && python3 seeded_identity.py --main $BASE --branch $ARM --root $W/identity --port $PORT > $W/identity.txt 2>&1)
echo "identity rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $K/startup_20261031 && timeout 1500 python3 mixed_long.py --a base=$BASE --b step=$ARM --harness $HARNESS --root $W/mixed --port $((PORT + 400)) > $W/mixed.txt 2>&1)
echo "mixed rc $? $(date +%T)" >> $L
grep -E ' card [01]: stream done|^  [a-z]+: [0-9]+ of |Traceback|Error' $W/mixed.txt | cut -c1-290 >> $L
port=$((PORT + 1400))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5 concurrency=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $concurrency --no-stream --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
for concurrency in 64 16; do
  run_arm base_n${concurrency}_o0 $BASE 0 $port $W/barrier_n${concurrency}_o0 $concurrency &
  run_arm step_n${concurrency}_o0 $ARM 1 $((port + 400)) $W/barrier_n${concurrency}_o0 $concurrency &
  wait
  port=$((port + 800))
  run_arm base_n${concurrency}_o1 $BASE 1 $port $W/barrier_n${concurrency}_o1 $concurrency &
  run_arm step_n${concurrency}_o1 $ARM 0 $((port + 400)) $W/barrier_n${concurrency}_o1 $concurrency &
  wait
  port=$((port + 800))
done
echo "queue $BLOCK done $(date +%T)" >> $L
