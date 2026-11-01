#!/usr/bin/env bash
# Block 59 of 11-01: the three PRs as they now stand (#3403 narrowed to the opaque ops, each branch with upstream main
# merged) held to byte identity against that main: seeded_identity.py per PR, concurrency 1, seed 1234, non-streaming
# then streaming, every request's audio by SHA-256. Then main against main on the standard SeedTTS A/B (ab_runs.py),
# the same-node band every PR's rows are read against.
# usage: D=... T=<kit tree> MAIN=<tree> CODEC=<tree> IMPORTS=<tree> WARMUPS=<tree> bash queue_block59.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
CODEC=${CODEC:?the codec PR tree}
IMPORTS=${IMPORTS:?the imports PR tree}
WARMUPS=${WARMUPS:?the warm-ups PR tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block59
L=$W/queue.log
mkdir -p $W
for arm in MAIN CODEC IMPORTS WARMUPS; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block59 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
port=26000
for arm in CODEC IMPORTS WARMUPS; do
  (cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch ${!arm} --root $W/identity_${arm,,} --port $port > $W/identity_${arm,,}.txt 2>&1)
  echo "${arm,,}: $(grep -E 'byte identical|error lines' $W/identity_${arm,,}.txt | tr '\n' '#')" >> $L
  port=$((port + 1000))
done
(cd $K/next_20261031 && python3 ab_runs.py --a maina=$MAIN --b mainb=$MAIN --root $W/band --port 29000 > $W/band.txt 2>&1)
echo "band: $(grep -E '^(n16|s16)_o[12]|paired' $W/band.txt | cut -c1-400 | tr '\n' '#')" >> $L
echo "queue block59 done $(date +%T)" >> $L
