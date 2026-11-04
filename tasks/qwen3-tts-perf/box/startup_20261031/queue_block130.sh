#!/usr/bin/env bash
# Block 130 of 11-04: the review's findings on #3477. Each new test on the PR's source, then on the fixes. The suite. The real codec
# with a replay failed partway, both trees. Seeded c1 identity against the PR. n64 same time, both orders. First chunks beside 64 long
# clients, where whole requests reach the queue in bursts past a batch, both orders.
# usage: D=... T=<kit tree> BASE=<PR tree> ARM=<fix tree> HARNESS=<tree> [BLOCK=block130] [PORT=12000] bash queue_block130.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the PR tree}
ARM=${ARM:?the fix tree}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block130}
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
NEW='failed_later_replay or behind_more_whole_requests or counts_only_first_chunks or aborted_in_flight or aborted_while_queued'
V=sglang_omni/models/qwen3_tts/streaming_vocoder.py
cp $ARM/$V $W/fixed_vocoder.py
git -C $ARM show $(git -C $BASE rev-parse HEAD):$V > $ARM/$V
(cd $ARM && CUDA_VISIBLE_DEVICES="" timeout 600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py -k "$NEW" > $W/new_tests_before.log 2>&1)
echo "new tests on the PR's source rc $?: $(grep -E '[0-9]+ (passed|failed)' $W/new_tests_before.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED" $W/new_tests_before.log | cut -c1-170 >> $L
cp $W/fixed_vocoder.py $ARM/$V
git -C $ARM diff --quiet HEAD -- sglang_omni && echo "source restored" >> $L || echo "SOURCE NOT RESTORED" >> $L
(cd $ARM && CUDA_VISIBLE_DEVICES="" timeout 600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py -k "$NEW" > $W/new_tests_after.log 2>&1)
echo "new tests on the fixes rc $?: $(grep -E '[0-9]+ (passed|failed)' $W/new_tests_after.log | tail -n 1 | cut -c1-90)" >> $L
(cd $ARM && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
for arm in BASE ARM; do
  tree=${!arm}
  (cd $tree && PYTHONPATH=$tree CUDA_VISIBLE_DEVICES=0 timeout 1500 python3 $K/startup_20261031/whole_fault_read.py > $W/fault_$arm.log 2>&1)
  echo "fault $arm rc $? $(date +%T)" >> $L
  grep -E '^\{"generated"' $W/fault_$arm.log | cut -c1-420 >> $L
done
(cd $K/next_20261031 && python3 seeded_identity.py --main $BASE --branch $ARM --root $W/identity --port $PORT > $W/identity.txt 2>&1)
echo "identity rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
port=$((PORT + 1000))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py --label $label --tree $tree --card $card --port $port --concurrency 64 --no-stream --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Failed requests|RTF mean|Throughput \(req/s\)|Latency p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log 2>/dev/null) left context $(grep -c left-context $W/arms/$label/serve.log 2>/dev/null)" >> $L
}
run_arm base_n64_o0 $BASE 0 $port $W/barrier_o0 &
run_arm fix_n64_o0 $ARM 1 $((port + 400)) $W/barrier_o0 &
wait
port=$((port + 800))
run_arm base_n64_o1 $BASE 1 $port $W/barrier_o1 &
run_arm fix_n64_o1 $ARM 0 $((port + 400)) $W/barrier_o1 &
wait
port=$((port + 800))
(cd $K/startup_20261031 && timeout 2400 python3 mixed_long.py --a base=$BASE --b fix=$ARM --harness $HARNESS --root $W/mixed --port $port --long-concurrency 64 --long-requests 192 > $W/mixed.txt 2>&1)
echo "mixed rc $? $(date +%T)" >> $L
grep -vE "^(o1|o2) (base|fix): card" $W/mixed.txt | cut -c1-300 >> $L
echo "queue $BLOCK done $(date +%T)" >> $L
