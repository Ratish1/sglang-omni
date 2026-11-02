#!/usr/bin/env bash
# Block 107 of 11-02: long streaming requests timed by a client that only counts bytes, because the stress harness's own
# loop stalls at this rate. Base against the arm, same time: c64 with 128 requests both card orders, c1 with 6 both orders.
# usage: D=... T=<kit tree> BASE=<tree> ARM=<tree> HARNESS=<tree> [BLOCK=block107] [CARD_A=2] [CARD_B=3] [PORT=38000] bash queue_block107.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the base tree}
ARM=${ARM:?the tree under test}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block107}
CARD_A=${CARD_A:-2}
CARD_B=${CARD_B:-3}
port=${PORT:-38000}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in BASE ARM; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block96/cache/inductor TRITON_CACHE_DIR=$D/block96/cache/triton
one() {
  local label=$1 tree=$2 card=$3 port=$4 concurrency=$5 requests=$6 barrier=$7
  (cd $K && timeout 2400 python3 long_stream_client.py --label $label --tree $tree --card $card --port $port \
    --concurrency $concurrency --requests $requests --harness $HARNESS --out $W/$label --barrier $barrier > $W/$label.txt 2>&1)
  echo "$label card $card exit $? $(date +%T); $(grep -E '^[a-z_0-9]+: [0-9]+ of ' $W/$label.txt | cut -c1-260)" >> $L
}
pair() {
  local name=$1 concurrency=$2 requests=$3 first=$4 second=$5
  one base_$name $BASE $first $port $concurrency $requests $W/barrier_$name &
  one arm_$name $ARM $second $((port + 400)) $concurrency $requests $W/barrier_$name &
  wait
  port=$((port + 800))
}
pair c64_o0 64 128 $CARD_A $CARD_B
pair c64_o1 64 128 $CARD_B $CARD_A
pair c1_o0 1 6 $CARD_A $CARD_B
pair c1_o1 1 6 $CARD_B $CARD_A
echo "queue $BLOCK done $(date +%T)" >> $L
