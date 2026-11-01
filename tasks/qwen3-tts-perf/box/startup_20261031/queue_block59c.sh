#!/usr/bin/env bash
# Block 59c of 11-01: block 59's warm-up pair never ran (its main server found port 28000 taken, 1,000 above the
# imports pair's), so it runs again on ports far from every other pair. Then the reference encoder with Inductor's
# deterministic option in its compile options (refenc_codes.py --deterministic): card 0 into two fresh caches, card 1
# into a fresh cache, compared with block 59b's runs (compiled on each card and cache, eager). Waits for block 59b.
# usage: D=... T=<kit tree> MAIN=<tree> WARMUPS=<tree> bash queue_block59c.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
WARMUPS=${WARMUPS:?the warm-ups PR tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block59c
B=$D/block59b
L=$W/queue.log
mkdir -p $W
until grep -q "queue block59b done" $B/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN WARMUPS; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block59c start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $MAIN --branch $WARMUPS --root $W/identity_warmups --port 44000 > $W/identity_warmups.txt 2>&1)
echo "warmups: $(grep -E 'byte identical|error lines|Error' $W/identity_warmups.txt | cut -c1-200 | tr '\n' '#')" >> $L
probe() {
  local name=$1 gpu=$2 cache=$3
  (cd $MAIN && HF_HUB_OFFLINE=1 PYTHONPATH=$MAIN TORCHINDUCTOR_CACHE_DIR=$W/$cache/inductor TRITON_CACHE_DIR=$W/$cache/triton \
    timeout 1200 python3 $K/next_20261031/refenc_codes.py --gpu $gpu --out $W/$name --deterministic > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T): $(tail -n 1 $W/$name.log)" >> $L
}
probe det_card0_cacheD 0 cacheD
probe det_card0_cacheF 0 cacheF
probe det_card1_cacheG 1 cacheG
(cd $MAIN && HF_HUB_OFFLINE=1 PYTHONPATH=$MAIN python3 $K/next_20261031/refenc_codes.py --compare $W/det_card0_cacheD.pt \
  $W/det_card0_cacheF.pt $W/det_card1_cacheG.pt $B/card0_cacheA.pt $B/card1_cacheC.pt $B/card0_eager.pt $B/card1_eager.pt \
  > $W/compare.txt 2>&1)
grep -vE "INFO|WARN|Using the latest|Found the latest" $W/compare.txt >> $L
echo "queue block59c done $(date +%T)" >> $L
