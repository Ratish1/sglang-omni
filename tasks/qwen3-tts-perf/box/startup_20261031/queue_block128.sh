#!/usr/bin/env bash
# Block 128 of 11-04: does the 32 row whole-request slot limit bind, and what does 64 change. One probe tree, 32 against 64 at the
# same time on two cards, both orders. SeedTTS non-streaming at c64 and c256; then SeedTTS non-streaming at 8 in flight beside 64
# clients of long requests, the load where more than 32 long rows hold slots. Occupancy comes from the probe's last WHOLE_SLOTS line.
# usage: D=... T=<kit tree> ARM=<probe tree> HARNESS=<tree> [BLOCK=block128] [PORT=3000] bash queue_block128.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
ARM=${ARM:?the probe tree}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block128}
PORT=${PORT:-3000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) probe $(git -C $ARM rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $ARM diff --quiet HEAD -- sglang_omni || echo DIRTY); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
port=$PORT
run_arm() {
  local label=$1 limit=$2 card=$3 port=$4 barrier=$5 concurrency=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $ARM --card $card --port $port --concurrency $concurrency --no-stream --barrier $barrier --parties 2 \
    "--serve-args=--vocoder.factory.initial_max_batch_size $limit --tts_engine.engine.max_queued_requests $concurrency" > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency (mean|p95|p99)' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-210); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
  grep WHOLE_SLOTS $W/arms/$label/serve.log | tail -n 1 | sed -E 's/.*WHOLE_SLOTS/  slots/' | cut -c1-900 >> $L
}
for concurrency in 64 256; do
  run_arm s32_n${concurrency}_o0 32 0 $port $W/barrier_n${concurrency}_o0 $concurrency &
  run_arm s64_n${concurrency}_o0 64 1 $((port + 400)) $W/barrier_n${concurrency}_o0 $concurrency &
  wait
  port=$((port + 800))
  run_arm s32_n${concurrency}_o1 32 1 $port $W/barrier_n${concurrency}_o1 $concurrency &
  run_arm s64_n${concurrency}_o1 64 0 $((port + 400)) $W/barrier_n${concurrency}_o1 $concurrency &
  wait
  port=$((port + 800))
done
(cd $K/startup_20261031 && timeout 2400 python3 mixed_long.py --a s32=$ARM --b s64=$ARM --harness $HARNESS --root $W/mixed --port $port \
  --short-no-stream --long-concurrency 64 --long-requests 192 \
  "--a-serve-args=--vocoder.factory.initial_max_batch_size 32" "--b-serve-args=--vocoder.factory.initial_max_batch_size 64" > $W/mixed.txt 2>&1)
echo "mixed rc $? $(date +%T)" >> $L
grep -vE "^(o1|o2) s(32|64): card" $W/mixed.txt | cut -c1-300 >> $L
for f in $W/mixed/o*/serve_s*.log; do echo "  $f $(grep WHOLE_SLOTS $f | tail -n 1 | sed -E 's/.*WHOLE_SLOTS//' | cut -c1-900)" >> $L; done
echo "queue $BLOCK done $(date +%T)" >> $L
