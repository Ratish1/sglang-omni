#!/usr/bin/env bash
# Block 58 of 11-01: #3404 (unused imports) and #3405 (graph warm-ups) held to byte identity like #3403 in block 57:
# seeded_identity.py against main, concurrency 1, seed 1234, non-streaming then streaming, every request's audio by
# SHA-256. Waits for block 57 first.
# usage: D=... T=<kit tree> MAIN=<tree> IMPORTS=<tree> WARMUPS=<tree> bash queue_block58.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
IMPORTS=${IMPORTS:?the imports tree}
WARMUPS=${WARMUPS:?the warm-ups tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block58
L=$W/queue.log
mkdir -p $W
until grep -q "queue block57 done" $D/block57/queue.log 2> /dev/null; do sleep 20; done
for tree in $MAIN $IMPORTS $WARMUPS; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block58 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
for arm in IMPORTS WARMUPS; do
  (cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch ${!arm} --root $W/${arm,,} --port 25000 > $W/${arm,,}.txt 2>&1)
  echo "${arm,,}: $(grep -E 'byte identical' $W/${arm,,}.txt | tr '\n' '#')" >> $L
done
echo "queue block58 done $(date +%T)" >> $L
