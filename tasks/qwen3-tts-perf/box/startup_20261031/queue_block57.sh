#!/usr/bin/env bash
# Block 57 of 11-01: #3403 held to byte identity. seeded_identity.py serves main (card 0) and the codec's opaque-ops
# head (card 1) at concurrency 1, seed 1234, non-streaming then streaming, all 1,088 SeedTTS EN prompts, and compares
# every request's audio by SHA-256. Waits for block 56b first.
# usage: D=... T=<kit tree> MAIN=<tree> BRANCH=<tree> bash queue_block57.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
BRANCH=${BRANCH:?the branch tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block57
L=$W/queue.log
mkdir -p $W
until grep -q "queue block56b done" $D/block56b/queue.log 2> /dev/null; do sleep 10; done
for tree in $MAIN $BRANCH; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block57 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch $BRANCH --root $W/identity --port 24000 > $W/identity.txt 2>&1)
grep -E "byte identical|healthy|error lines" $W/identity.txt >> $L
echo "queue block57 done $(date +%T)" >> $L
