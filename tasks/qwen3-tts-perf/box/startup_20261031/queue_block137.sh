#!/usr/bin/env bash
# Block 137 of 11-05: whole requests as rows with the graph teardown merged. The suite. Real codec with a replay failed: rows in the
# failed replay error, the rest finish, each teardown case in its own process. A real server through a failed replay. Then what the
# row slots buy: as many slots as one replay takes against 64, with non-streaming c64 and with short requests beside 64 long ones,
# both card orders, each server on its own socket.
# usage: D=... T=<kit tree> ARM=<tree> EIGHT=<tree> HARNESS=<tree> [BLOCK=block137] [PORT=7000] bash queue_block137.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
ARM=${ARM:?the rows tree}
EIGHT=${EIGHT:?the rows tree with eight slots}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block137}
PORT=${PORT:-7000}
K=$T/tasks/qwen3-tts-perf/box
S=$K/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
LARGE=Qwen/Qwen3-TTS-12Hz-1.7B-Base
export CARD_CPUS="0=0-31,64-95;1=32-63,96-127"
mkdir -p $W
for arm in ARM EIGHT; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $ARM && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline/test_stage.py > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
for cases in "0 1 2" 3 4; do
  name=fault_$(echo $cases | tr -d ' ')
  (cd $ARM && PYTHONPATH=$ARM CUDA_VISIBLE_DEVICES=0 timeout 1500 python3 $S/whole_fault_read.py --only-case $cases > $W/$name.log 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E '^\{"(stream_slots|case)"' $W/$name.log | cut -c1-280 >> $L
done
(cd $HARNESS && timeout 1500 python3 $S/graph_fault_read.py --tree $ARM --harness $HARNESS --card 0 --port $PORT --mode window --replay 3 --out $W/rows_fault_window > $W/rows_fault_window.txt 2>&1) &
(cd $HARNESS && timeout 1500 python3 $S/graph_fault_read.py --tree $ARM --harness $HARNESS --card 1 --port $((PORT + 300)) --mode tail --replay 3 --out $W/rows_fault_tail > $W/rows_fault_tail.txt 2>&1) &
wait
for mode in window tail; do
  echo "rows_fault_$mode $(date +%T)" >> $L
  grep -aE '^(ROUND|FAULT) ' $W/rows_fault_$mode.txt | cut -c1-1200 >> $L
done
port=$((PORT + 600))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py --label $label --tree $tree --card $card --port $port --concurrency 64 --no-stream --model $LARGE --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency mean|Latency p50|Latency p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-220); memory $(cut -c1-60 $W/arms/$label/memory.txt 2>/dev/null); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
for order in o0 o1; do
  if [ $order = o0 ]; then first=0; second=1; else first=1; second=0; fi
  run_arm slots64_n64_$order $ARM $first $port $W/barrier_$order &
  run_arm slots8_n64_$order $EIGHT $second $((port + 400)) $W/barrier_$order &
  wait
  port=$((port + 800))
done
(cd $S && timeout 2400 python3 mixed_long.py --a slots64=$ARM --b slots8=$EIGHT --harness $HARNESS --root $W/mixed --port $port --short-no-stream --long-concurrency 64 --long-requests 192 > $W/mixed.txt 2>&1)
echo "mixed rc $? $(date +%T)" >> $L
grep -vE "^(o1|o2) (slots64|slots8): card" $W/mixed.txt | cut -c1-300 >> $L
echo "queue $BLOCK done $(date +%T)" >> $L
