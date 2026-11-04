#!/usr/bin/env bash
# Block 122 of 11-04: the replay-at-a-time tree's last reads. The touched suite on the fix. What remains of the first chunk tail
# beside long requests, by gap. Short non-streaming requests beside long ones, against the 256 tree and main. Then main on
# 0.5.23 against it, the PR's tables: first chunks beside long requests, SeedTTS non-streaming c64 and c16, streaming c64.
# usage: D=... T=<kit tree> MAIN=<tree> BASE=<tree> ARM=<tree> HARNESS=<tree> [BLOCK=block122] [PORT=14000] bash queue_block122.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
BASE=${BASE:?the 256 tree}
ARM=${ARM:?the tree under test}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block122}
PORT=${PORT:-14000}
K=$T/tasks/qwen3-tts-perf/box
S=$K/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in MAIN BASE ARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $ARM && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
tail_run() {
  local name=$1 card=$2 port=$3
  (cd $S && timeout 1800 python3 first_chunk_tail.py --tree $ARM --card $card --port $port --root $W/$name --harness $HARNESS --long-requests 32 > $W/$name.txt 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E '^stream done|^ *long: |Traceback|Error' $W/$name.txt | cut -c1-260 >> $L
}
tail_run tail0 0 $PORT &
tail_run tail1 1 $((PORT + 400)) &
wait
mixed() {
  local name=$1 first=$2 second=$3 port=$4 extra=$5
  (cd $S && timeout 1500 python3 mixed_long.py --a ${first%%=*}=${first#*=} --b ${second%%=*}=${second#*=} --harness $HARNESS --root $W/$name --port $port $extra > $W/$name.txt 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E ' card [01]: (stream|short) done|^  [a-z]+: [0-9]+ of |Traceback|Error' $W/$name.txt | cut -c1-290 >> $L
}
mixed short_base base=$BASE step=$ARM $((PORT + 800)) --short-no-stream
mixed short_main main=$MAIN step=$ARM $((PORT + 1600)) --short-no-stream
mixed stream_main main=$MAIN step=$ARM $((PORT + 2400)) ""
port=$((PORT + 3200))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5 concurrency=$6 mode=$7
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $concurrency $mode --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95|TTFC mean|TTFC p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-210); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
pair() {
  local name=$1 concurrency=$2 mode=$3
  run_arm main_${name}_o0 $MAIN 0 $port $W/barrier_${name}_o0 $concurrency "$mode" &
  run_arm step_${name}_o0 $ARM 1 $((port + 400)) $W/barrier_${name}_o0 $concurrency "$mode" &
  wait
  port=$((port + 800))
  run_arm main_${name}_o1 $MAIN 1 $port $W/barrier_${name}_o1 $concurrency "$mode" &
  run_arm step_${name}_o1 $ARM 0 $((port + 400)) $W/barrier_${name}_o1 $concurrency "$mode" &
  wait
  port=$((port + 800))
}
pair n64 64 --no-stream
pair n16 16 --no-stream
pair s64 64 ""
echo "queue $BLOCK done $(date +%T)" >> $L
