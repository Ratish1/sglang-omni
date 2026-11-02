#!/usr/bin/env bash
# Block 113 of 11-02: is c64 bimodal by boot? Block 111 read 67.45 req/s on the host path head where 11 other boots read about 78,
# and that is the baseline's level, so the +15 % needs a boot distribution before it stands. Independent boots of the base and the
# head, same time, swapping cards each round, SeedTTS EN streaming at 64, no scoring. Each boot's request records stay for a
# by-state read of any slow boot. Rule: the gain stands if every head boot beats every base boot, or the slow fraction is read and
# reported with the mode of each arm.
# usage: D=... T=<kit tree> BASE=<tree> ARM=<tree> [ROUNDS=4] [BLOCK=block113] [CARD_A=0] [CARD_B=1] [PORT=54000] bash queue_block113.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the base tree}
ARM=${ARM:?the tree under test}
ROUNDS=${ROUNDS:-4}
BLOCK=${BLOCK:-block113}
CARD_A=${CARD_A:-0}
CARD_B=${CARD_B:-1}
port=${PORT:-54000}
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
run_arm() {
  local label=$1 tree=$2 card=$3 port=$4 barrier=$5
  (cd $K/inline_20261026 && ARM_ROOT=$W/arms timeout 1800 python3 serve_arm.py \
    --label $label --tree $tree --card $card --port $port --concurrency 64 --barrier $barrier --parties 2 > $W/$label.out 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -hE 'RTF mean|Throughput \(req/s\)|TTFC mean|TTFC p95' $W/arms/$label/bench.log | tr -s ' ' | tr '\n' ';' | cut -c1-150); tracebacks $(grep -c Traceback $W/arms/$label/serve.log)" >> $L
}
for round in $(seq 1 $ROUNDS); do
  run_arm base_r${round}a $BASE $CARD_A $port $W/barrier_r${round}a &
  run_arm arm_r${round}a $ARM $CARD_B $((port + 400)) $W/barrier_r${round}a &
  wait
  port=$((port + 800))
  run_arm base_r${round}b $BASE $CARD_B $port $W/barrier_r${round}b &
  run_arm arm_r${round}b $ARM $CARD_A $((port + 400)) $W/barrier_r${round}b &
  wait
  port=$((port + 800))
done
echo "queue $BLOCK done $(date +%T)" >> $L
