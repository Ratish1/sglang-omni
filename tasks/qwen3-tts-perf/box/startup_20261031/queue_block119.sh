#!/usr/bin/env bash
# Block 119 of 11-04: is a whole request's 256-frame piece needed, and at what length. SeedTTS's longest request generates 116
# frames, so only long requests can show it: 8 streaming SeedTTS clients beside 8 non-streaming 2,048-frame clients, both
# card orders. An A/A first gives the first chunk tail its band, then 256 against one widest window (64) and against no
# bound. A piece of 64 also reaches SeedTTS (204 of 1,088 requests generate more), so that arm reads non-streaming c64 too.
# usage: D=... T=<kit tree> BASE=<tree> SMALL=<tree> NONE=<tree> HARNESS=<tree> [BLOCK=block119] [PORT=10000] bash queue_block119.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the 256 tree}
SMALL=${SMALL:?the 64 tree}
NONE=${NONE:?the unbounded tree}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block119}
PORT=${PORT:-10000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in BASE SMALL NONE; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
mixed() {
  local name=$1 second=$2 port=$3
  (cd $K/startup_20261031 && timeout 1500 python3 mixed_long.py --a base=$BASE --b $name=$second --harness $HARNESS --root $W/$name --port $port > $W/$name.txt 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E ' card [01]: stream done|^  [a-z]+: [0-9]+ of |Traceback|Error' $W/$name.txt | cut -c1-290 >> $L
}
mixed same $BASE $PORT
mixed small $SMALL $((PORT + 1000))
mixed none $NONE $((PORT + 2000))
port=$((PORT + 3000))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 1500 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency 64 --no-stream --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log)" >> $L
}
run_arm base_n64_o0 $BASE 0 $port $W/barrier_o0 &
run_arm small_n64_o0 $SMALL 1 $((port + 400)) $W/barrier_o0 &
wait
port=$((port + 800))
run_arm base_n64_o1 $BASE 1 $port $W/barrier_o1 &
run_arm small_n64_o1 $SMALL 0 $((port + 400)) $W/barrier_o1 &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
