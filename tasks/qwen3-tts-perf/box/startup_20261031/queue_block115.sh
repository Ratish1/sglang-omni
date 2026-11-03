#!/usr/bin/env bash
# Block 115 of 11-03: the non-streaming PR's head with main merged, the last read before it opens. The PR's measured
# tables stand, so this reads only what the merge and the three small commits could change. The touched suite. Seeded
# concurrency 1 in both modes against the last measured head: no value moved, so anything but byte identical stops the PR.
# Non-streaming c64 against main, same time, both card orders: read against the PR table's 75.28, 75.40 req/s.
# usage: D=... T=<kit tree> LAST=<tree> MAIN=<tree> PR=<tree> [BLOCK=block115] [CARD_A=0] [CARD_B=1] [PORT=12000] bash queue_block115.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
LAST=${LAST:?the last measured head}
MAIN=${MAIN:?main}
PR=${PR:?the tree under test}
BLOCK=${BLOCK:-block115}
CARD_A=${CARD_A:-0}
CARD_B=${CARD_B:-1}
PORT=${PORT:-12000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in LAST MAIN PR; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $PR && TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 2400 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1 | cut -c1-90)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 16 | cut -c1-170 >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
(cd $K/next_20261031 && python3 seeded_identity.py --main $LAST --branch $PR --root $W/identity --port $PORT > $W/identity.txt 2>&1)
echo "identity rc $? $(date +%T): $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
port=$((PORT + 400))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency 64 --no-stream --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T) startup $(cat $W/arms/$label/startup.txt 2>/dev/null); $(grep -hE 'Completed requests|Failed requests|RTF mean|Throughput \(req/s\)|Latency mean|Latency p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-190); tracebacks $(grep -c Traceback $W/arms/$label/serve.log)" >> $L
}
run_arm main_o0 $MAIN $CARD_A $port $W/barrier_o0 &
run_arm pr_o0 $PR $CARD_B $((port + 400)) $W/barrier_o0 &
wait
port=$((port + 800))
run_arm main_o1 $MAIN $CARD_B $port $W/barrier_o1 &
run_arm pr_o1 $PR $CARD_A $((port + 400)) $W/barrier_o1 &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
