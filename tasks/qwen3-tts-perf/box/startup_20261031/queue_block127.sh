#!/usr/bin/env bash
# Block 127 of 11-04: the exact head against main for the PR's top table, non-streaming c64 and c16, both orders.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> [BLOCK=block127] [PORT=2050] bash queue_block127.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the head}
BLOCK=${BLOCK:-block127}
PORT=${PORT:-2050}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ') head $(git -C $ARM rev-parse HEAD HEAD:sglang_omni | tr '\n' ' '); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
port=$PORT
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5 concurrency=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $concurrency --no-stream --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
for concurrency in 64 16; do
  run_arm main_n${concurrency}_o0 $MAIN 0 $port $W/barrier_n${concurrency}_o0 $concurrency &
  run_arm head_n${concurrency}_o0 $ARM 1 $((port + 400)) $W/barrier_n${concurrency}_o0 $concurrency &
  wait
  port=$((port + 800))
  run_arm main_n${concurrency}_o1 $MAIN 1 $port $W/barrier_n${concurrency}_o1 $concurrency &
  run_arm head_n${concurrency}_o1 $ARM 0 $((port + 400)) $W/barrier_n${concurrency}_o1 $concurrency &
  wait
  port=$((port + 800))
done
echo "queue $BLOCK done $(date +%T)" >> $L
