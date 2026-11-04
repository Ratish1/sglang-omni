#!/usr/bin/env bash
# Block 126 of 11-04: the review's fixes on the measured head. A worker with replays due took one queued entry a pass, so
# first chunks and new rows entered one replay apart: it now takes all that has queued. Touched suite; seeded c1 bytes against
# the measured head; non-streaming c64 and first chunks beside long requests against main, both orders; the tail by gap.
# usage: D=... T=<kit tree> MAIN=<tree> LAST=<tree> ARM=<tree> HARNESS=<tree> [BLOCK=block126] [PORT=2050] bash queue_block126.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
LAST=${LAST:?the measured head}
ARM=${ARM:?the head}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block126}
PORT=${PORT:-2050}
K=$T/tasks/qwen3-tts-perf/box
S=$K/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in MAIN LAST ARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $ARM && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline/test_stage.py > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
port=$PORT
(cd $K/next_20261031 && python3 seeded_identity.py --main $LAST --branch $ARM --root $W/identity --port $port > $W/identity.txt 2>&1)
echo "identity rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
port=$((port + 800))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency 64 --no-stream --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
run_arm last_n64_o0 $LAST 0 $port $W/barrier_o0 &
run_arm head_n64_o0 $ARM 1 $((port + 400)) $W/barrier_o0 &
wait
port=$((port + 800))
run_arm last_n64_o1 $LAST 1 $port $W/barrier_o1 &
run_arm head_n64_o1 $ARM 0 $((port + 400)) $W/barrier_o1 &
wait
port=$((port + 800))
(cd $S && timeout 1500 python3 mixed_long.py --a main=$MAIN --b head=$ARM --harness $HARNESS --root $W/mixed --port $port > $W/mixed.txt 2>&1)
echo "mixed rc $? $(date +%T)" >> $L
grep -E ' card [01]: stream done|^  [a-z]+: [0-9]+ of |Traceback|Error' $W/mixed.txt | cut -c1-290 >> $L
port=$((port + 800))
tail_run() {
  local name=$1 card=$2 port=$3
  (cd $S && timeout 1800 python3 first_chunk_tail.py --tree $ARM --card $card --port $port --root $W/$name --harness $HARNESS --long-requests 32 > $W/$name.txt 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E '^stream done|^ *long: |Traceback|Error' $W/$name.txt | cut -c1-260 >> $L
}
tail_run tail0 0 $port &
tail_run tail1 1 $((port + 400)) &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
