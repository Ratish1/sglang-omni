#!/usr/bin/env bash
# Block 59e of 11-01: the reference encoder's codes move with one kernel's first-launch pick (block 59b: the split
# LayerNorm's final welford combine, a persistent reduction over 4 partials with several XBLOCK configs). Inductor keeps
# one persistent-reduction config for an INNER reduction of 256 or more elements, so without the split the LayerNorm
# should autotune nothing. Each variant on card 0 into two fresh caches and card 1 into one (refenc_codes.py --option):
# split_reductions=False, then triton.persistent_reductions=False; codes compared with each other, with block 59b's
# compiled states and eager, and the encode time. Waits for block 59c.
# usage: D=... T=<kit tree> MAIN=<tree> bash queue_block59e.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block59e
B=$D/block59b
L=$W/queue.log
mkdir -p $W
until grep -q "queue block59c done" $D/block59c/queue.log 2> /dev/null; do sleep 20; done
echo "tree MAIN $MAIN $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $MAIN diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block59e start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
probe() {
  local name=$1 gpu=$2 cache=$3 option=$4
  (cd $MAIN && HF_HUB_OFFLINE=1 PYTHONPATH=$MAIN TORCHINDUCTOR_CACHE_DIR=$W/$cache/inductor TRITON_CACHE_DIR=$W/$cache/triton \
    timeout 1200 python3 $K/next_20261031/refenc_codes.py --gpu $gpu --out $W/$name --option $option > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T): $(tail -n 1 $W/$name.log)" >> $L
}
probe nosplit_card0_a 0 cacheNa split_reductions=False
probe nosplit_card0_b 0 cacheNb split_reductions=False
probe nosplit_card1_c 1 cacheNc split_reductions=False
probe nopersist_card0_a 0 cachePa triton.persistent_reductions=False
probe nopersist_card0_b 0 cachePb triton.persistent_reductions=False
probe nopersist_card1_c 1 cachePc triton.persistent_reductions=False
(cd $MAIN && HF_HUB_OFFLINE=1 PYTHONPATH=$MAIN python3 $K/next_20261031/refenc_codes.py --compare $W/nosplit_card0_a.pt \
  $W/nosplit_card0_b.pt $W/nosplit_card1_c.pt $W/nopersist_card0_a.pt $W/nopersist_card0_b.pt $W/nopersist_card1_c.pt \
  $B/card0_cacheA.pt $B/card0_cacheB.pt $B/card0_eager.pt > $W/compare.txt 2>&1)
grep -vE "INFO|WARN|Using the latest|Found the latest" $W/compare.txt >> $L
echo "queue block59e done $(date +%T)" >> $L
