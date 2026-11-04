#!/usr/bin/env bash
# Block 123 of 11-04: two steps on the replay-at-a-time tree, each read alone. A whole request's first decode through the tail
# graphs with only its last window rounded (63 % of its frames were reference or padding): non-streaming c64 and c16 both
# orders, then seeded quality on the same codes, since the first decode's arithmetic changes. One buffer the result carries,
# no samples copied when a long request ends (about 86 ms on the first-audio worker for 8 rows of 2,048 frames): byte
# identity against the tail tree, the first chunk tail by gap, then first chunks beside long requests against the replay tree.
# usage: D=... T=<kit tree> BASE=<tree> ARM=<tail tree> BUFFER=<tree> HARNESS=<tree> [BLOCK=block123] [PORT=20000] bash queue_block123.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the replay tree}
ARM=${ARM:?the tail tree}
BUFFER=${BUFFER:?the buffer tree}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block123}
PORT=${PORT:-20000}
K=$T/tasks/qwen3-tts-perf/box
S=$K/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in BASE ARM BUFFER; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $BUFFER && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
port=$PORT
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5 concurrency=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $concurrency --no-stream --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
for concurrency in 64 16; do
  run_arm step_n${concurrency}_o0 $BASE 0 $port $W/barrier_n${concurrency}_o0 $concurrency &
  run_arm tail_n${concurrency}_o0 $ARM 1 $((port + 400)) $W/barrier_n${concurrency}_o0 $concurrency &
  wait
  port=$((port + 800))
  run_arm step_n${concurrency}_o1 $BASE 1 $port $W/barrier_n${concurrency}_o1 $concurrency &
  run_arm tail_n${concurrency}_o1 $ARM 0 $((port + 400)) $W/barrier_n${concurrency}_o1 $concurrency &
  wait
  port=$((port + 800))
done
(cd $S && timeout 3000 python3 whole_seeded_pair.py --base $BASE --arm $ARM --root $W/seeded --port $port --no-streaming > $W/seeded.txt 2>&1)
echo "seeded rc $? $(date +%T)" >> $L
grep -E 'SIM|WER|Error|Traceback|paired|byte identical|req/s|RTF' $W/seeded.txt | cut -c1-240 | head -n 24 >> $L
port=$((port + 800))
(cd $K/next_20261031 && python3 seeded_identity.py --main $ARM --branch $BUFFER --root $W/identity --port $port > $W/identity.txt 2>&1)
echo "identity rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
port=$((port + 800))
tail_run() {
  local name=$1 card=$2 port=$3
  (cd $S && timeout 1800 python3 first_chunk_tail.py --tree $BUFFER --card $card --port $port --root $W/$name --harness $HARNESS --long-requests 32 > $W/$name.txt 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E '^stream done|^ *long: |Traceback|Error' $W/$name.txt | cut -c1-260 >> $L
}
tail_run tail0 0 $port &
tail_run tail1 1 $((port + 400)) &
wait
port=$((port + 800))
(cd $S && timeout 1500 python3 mixed_long.py --a step=$BASE --b buffer=$BUFFER --harness $HARNESS --root $W/mixed --port $port > $W/mixed.txt 2>&1)
echo "mixed rc $? $(date +%T)" >> $L
grep -E ' card [01]: (stream|short) done|^  [a-z]+: [0-9]+ of |Traceback|Error' $W/mixed.txt | cut -c1-290 >> $L
echo "queue $BLOCK done $(date +%T)" >> $L
