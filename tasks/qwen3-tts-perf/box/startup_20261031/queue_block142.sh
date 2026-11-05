#!/usr/bin/env bash
# Block 142 of 11-05: where the rows head's first chunk leans at streaming c64, by request state. Paired by prompt it reads +1.82 +- .67
# ms with p50 equal, where the teardown's near A/A reads -.52 +- .62. Same-time pairs with request events on, both orders, twice.
# usage: D=... T=<kit tree> MAIN=<tree> ARM=<tree> HARNESS=<tree> [BLOCK=block142] bash queue_block142.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
ARM=${ARM:?the tree under test}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block142}
S=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
export CARD_CPUS="0=0-31,64-95;1=32-63,96-127"
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) main $(git -C $MAIN rev-parse HEAD) arm $(git -C $ARM rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
tail_arm() {
  local name=$1 tree=$2 card=$3 port=$4
  (cd $S && timeout 1800 python3 first_chunk_tail.py --tree $tree --card $card --port $port --root $W/$name --harness $HARNESS --concurrency 64 > $W/$name.txt 2>&1)
  echo "$name card $card rc $? $(date +%T)" >> $L
  grep -E "^stream done|^streams |written," $W/$name.txt | cut -c1-200 >> $L
}
tail_arm main_o0_r1 $MAIN 0 11000 &
tail_arm head_o0_r1 $ARM 1 12400 &
wait
tail_arm main_o1_r1 $MAIN 1 13800 &
tail_arm head_o1_r1 $ARM 0 15200 &
wait
tail_arm main_o0_r2 $MAIN 0 16600 &
tail_arm head_o0_r2 $ARM 1 18000 &
wait
tail_arm main_o1_r2 $MAIN 1 19400 &
tail_arm head_o1_r2 $ARM 0 20800 &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
