#!/usr/bin/env bash
# Block 118 of 11-03: what block 117 left. First chunks beside long non-streaming requests on 1.7B against main, both card
# orders, after the driver's media path fix. Then two new seeds of 0.6B non-streaming at concurrency 1 against main, scored:
# seed 1234 leaned SIM -0.015 (95 % -0.032 to +0.001) with WER better, and a lean on a quality row gets new draws until
# bounded. Seeds repeat across boots, so these are new seeds, not reruns.
# usage: D=... T=<kit tree> MAIN=<tree> NEW=<tree> HARNESS=<tree> [BLOCK=block118] [PORT=10000] bash queue_block118.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?main}
NEW=${NEW:?the tree under test}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block118}
PORT=${PORT:-10000}
K=$T/tasks/qwen3-tts-perf/box
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
for arm in MAIN NEW; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
(cd $K/startup_20261031 && timeout 3000 python3 mixed_long.py --a main=$MAIN --b new=$NEW --harness $HARNESS --root $W/mixed_long --port $PORT > $W/mixed_long.txt 2>&1)
echo "mixed long rc $? $(date +%T)" >> $L
grep -E ' card [01]: stream done|^  [a-z]+: [0-9]+ of |Traceback|Error' $W/mixed_long.txt | cut -c1-300 >> $L
export BOX_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
port=$((PORT + 2000))
for seed in 2345 3456; do
  (cd $K/startup_20261031 && timeout 3000 python3 whole_seeded_pair.py --base $MAIN --arm $NEW --root $W/small_seed_$seed --port $port --seed $seed --no-streaming > $W/small_seed_$seed.txt 2>&1)
  echo "0.6B seed $seed rc $? $(date +%T)" >> $L
  grep -E 'c1n SIM|c1n WER|error lines|Traceback|Error' $W/small_seed_$seed.txt | cut -c1-200 >> $L
  port=$((port + 1000))
done
echo "queue $BLOCK done $(date +%T)" >> $L
