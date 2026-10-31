#!/usr/bin/env bash
# Block 43b of 10-31: block 43's identity check redone with one arena slot per row (block 43 gave every row the
# scratch slot, whose ring rows the rows race on), on block 43's warm caches: main twice (the cross-process band),
# then ops, batch, all, main again; then the all form with pointwise autotuning off, cold then warm in its own cache.
# usage: D=... T=<kit tree> MAIN=<main tree> FORMS=<forms tree> bash queue_block43b.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
FORMS=${FORMS:?the forms tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/block43
O=$D/block43b
L=$O/queue.log
mkdir -p $O
for tree in $MAIN $FORMS; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block43b start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

run_once() {
  local name=$1 tree=$2 form=$3 cache=$4 extra=$5
  local began=$(date +%s.%N)
  (cd $tree && PYTHONPATH=$tree SGLANG_OMNI_CODEC_COMPILE_FORM=$form \
    TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    timeout 1800 python3 $K/codec_forms.py --out $O/$name $extra > $O/$name.log 2>&1)
  echo "$name exit $? wall $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T)" >> $L
}

run_once main_a_warm $MAIN static $W/cache_main ""
run_once main_b_warm $MAIN static $W/cache_main ""
run_once ops_warm $FORMS ops $W/cache_ops ""
run_once batch_warm $FORMS batch $W/cache_batch ""
run_once all_warm $FORMS all $W/cache_all ""
run_once main_c_warm $MAIN static $W/cache_main ""
rm -rf $O/cache_all_noap
run_once all_noap_cold $FORMS all $O/cache_all_noap "--no-autotune-pointwise"
run_once all_noap_warm $FORMS all $O/cache_all_noap "--no-autotune-pointwise"
python3 $K/compare_forms.py $O main_a main_b ops batch all main_c all_noap > $O/compare.txt 2>&1
echo "queue block43b done $(date +%T)" >> $L
