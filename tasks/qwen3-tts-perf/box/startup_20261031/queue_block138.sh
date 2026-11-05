#!/usr/bin/env bash
# Block 138 of 11-05: the graph teardown against main, the standard SeedTTS A/B a Qwen3-TTS PR takes though no healthy path changes.
# Seeded c1 bytes in both modes, streaming c64 and c16 and non-streaming c16, same-time pairs in both card orders on own sockets.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> [BLOCK=block138] [PORT=9000] bash queue_block138.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the tree under test}
BLOCK=${BLOCK:-block138}
PORT=${PORT:-9000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
LARGE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
export CARD_CPUS="0=0-31,64-95;1=32-63,96-127"
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
mkdir -p $W
for arm in MAIN ARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch $ARM --root $W/identity --port $PORT > $W/identity.txt 2>&1)
echo "identity rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
port=$((PORT + 800))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5 concurrency=$6 flags=$7
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py --label $label --tree $tree --card $card --port $port --concurrency $concurrency $flags --model $LARGE --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95|TTFC mean|TTFC p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-200); memory $(cut -c1-60 $W/arms/$label/memory.txt 2>/dev/null); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
pair() {
  local name=$1 concurrency=$2 flags=$3
  run_arm main_${name}_o0 $MAIN 0 $port $W/barrier_${name}_o0 $concurrency "$flags" &
  run_arm fix_${name}_o0 $ARM 1 $((port + 400)) $W/barrier_${name}_o0 $concurrency "$flags" &
  wait
  port=$((port + 800))
  run_arm main_${name}_o1 $MAIN 1 $port $W/barrier_${name}_o1 $concurrency "$flags" &
  run_arm fix_${name}_o1 $ARM 0 $((port + 400)) $W/barrier_${name}_o1 $concurrency "$flags" &
  wait
  port=$((port + 800))
}
pair s64 64 ""
pair s16 16 ""
pair n16 16 --no-stream
echo "queue $BLOCK done $(date +%T)" >> $L
