#!/usr/bin/env bash
# Block 100 of 11-02: the code table and step message against the chain base. Seeded concurrency 1, streaming and whole,
# every file by SHA-256: this slice moves no value, so anything but identical is a defect. Then the streaming SeedTTS EN
# pair at 64, 16 and 1, same time, both card orders, no scoring because the bytes settle quality. Keep if c64 gains at
# least 5 % req/s and c1 and c16 lose nothing; the slice is also the base of what follows, so a level read is not a stop.
# usage: D=... T=<kit tree> BASE=<tree> ARM=<tree> [W=block100] [CARDS="0 1"] bash queue_block100.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the base tree}
ARM=${ARM:?the tree under test}
BLOCK=${BLOCK:-block100}
CARD_A=${CARD_A:-0}
CARD_B=${CARD_B:-1}
PORT=${PORT:-24000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in BASE ARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block96/cache/inductor TRITON_CACHE_DIR=$D/block96/cache/triton
(cd $K/next_20261031 && CARD_MAIN=$CARD_A CARD_BRANCH=$CARD_B python3 seeded_identity.py --main $BASE --branch $ARM --root $W/identity --port $PORT > $W/identity.txt 2>&1)
echo "identity: $(grep -E 'byte identical|error lines|Error|Traceback' $W/identity.txt | cut -c1-140 | tr '\n' '#')" >> $L
port=$((PORT + 400))
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 conc=$5 barrier=$6
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 2400 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency $conc --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'RTF mean|Throughput \(req/s\)|TTFC mean|TTFC p50|TTFC p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log)" >> $L
}
for conc in 64 16 1; do
  run_arm base_c${conc}_o0 $BASE $CARD_A $port $conc $W/barrier_c${conc}_o0 &
  run_arm arm_c${conc}_o0 $ARM $CARD_B $((port + 400)) $conc $W/barrier_c${conc}_o0 &
  wait
  port=$((port + 800))
  run_arm base_c${conc}_o1 $BASE $CARD_B $port $conc $W/barrier_c${conc}_o1 &
  run_arm arm_c${conc}_o1 $ARM $CARD_A $((port + 400)) $conc $W/barrier_c${conc}_o1 &
  wait
  port=$((port + 800))
done
echo "queue $BLOCK done $(date +%T)" >> $L
