#!/usr/bin/env bash
# Block 97 of 11-02: evict-first weight loads in the predictor's fused kernels, a read of R2's first item. predictor_step_bench.py
# at batch 1, 16, 64 on the chain base against the variant, at the same time on cards 2 and 3, then swapped. Bit identical by
# construction: the bench's hash of codes and summed embeddings must match between arms.
# usage: D=... T=<kit tree> BASE=<tree> VARIANT=<tree> bash queue_block97.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the base tree}
VARIANT=${VARIANT:?the variant tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block97
L=$W/queue.log
mkdir -p $W
for arm in BASE VARIANT; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block97 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
bench() {
  local label=$1 tree=$2 card=$3
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_${label%%_*}/inductor TRITON_CACHE_DIR=$W/cache_${label%%_*}/triton timeout 1800 \
    python3 $K/chain_20261022/predictor_step_bench.py --batch 1 16 64 --replays 200 --card $card > $W/$label.log 2>&1)
  echo "$label card $card exit $? $(date +%T)" >> $L
}
bench base_o0 $BASE 2 &
bench variant_o0 $VARIANT 3 &
wait
bench base_o1 $BASE 3 &
bench variant_o1 $VARIANT 2 &
wait
for label in base_o0 variant_o0 base_o1 variant_o1; do
  echo "== $label" >> $L
  grep -E "batch|median|hash|replay|kernels per step" $W/$label.log | cut -c1-200 | head -n 14 >> $L
done
echo "queue block97 done $(date +%T)" >> $L
