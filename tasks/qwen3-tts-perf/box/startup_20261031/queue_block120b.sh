#!/usr/bin/env bash
# Block 120b: block 120's first chunk tail with every event kept: the default 1,024 event buffer dropped most of them, and
# a stop does not wait for the writers.
# usage: D=... T=<kit tree> TREE=<tree> HARNESS=<tree> [BLOCK=block120b] [PORT=11000] bash queue_block120b.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
TREE=${TREE:?the tree}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block120b}
PORT=${PORT:-11000}
S=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) tree $(git -C $TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $TREE diff --quiet HEAD -- sglang_omni || echo DIRTY); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
tail_run() {
  local name=$1 card=$2 port=$3 long=$4
  (cd $S && timeout 1800 python3 first_chunk_tail.py --tree $TREE --card $card --port $port --root $W/$name --harness $HARNESS --long-requests $long > $W/$name.txt 2>&1)
  echo "$name rc $? $(date +%T)" >> $L
  grep -E '^stream done|^ *long: |Traceback|Error' $W/$name.txt | cut -c1-260 >> $L
}
tail_run mixed 0 $PORT 32 &
tail_run floor 1 $((PORT + 400)) 0 &
wait
tail_run mixed2 1 $((PORT + 800)) 32 &
tail_run floor2 0 $((PORT + 1200)) 0 &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
