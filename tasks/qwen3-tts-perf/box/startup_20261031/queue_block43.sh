#!/usr/bin/env bash
# Block 43 of 10-31: the incremental codec's compile forms (startup_20261031/PLAN.md item 1), one arm at a time on one
# card, each arm cold (its own empty inductor and Triton caches) then warm (the same caches), each a fresh process
# running codec_forms.py. Arms: main; the forms branch with the four Triton kernels as opaque ops (ops), plus the batch
# dynamic (batch), plus the width dynamic (all), the last two again with size-oblivious batch and width (obl); main
# with pointwise autotuning off (main_noap). Then compare_forms.py against main.
# usage: D=... T=<kit tree> MAIN=<main tree> FORMS=<forms tree> bash queue_block43.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
FORMS=${FORMS:?the forms tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/block43
L=$W/queue.log
mkdir -p $W
for tree in $MAIN $FORMS; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block43 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

run_arm() {
  local arm=$1 tree=$2 form=$3 oblivious=$4 extra=$5
  rm -rf $W/cache_$arm
  for phase in cold warm; do
    local began=$(date +%s.%N)
    (cd $tree && PYTHONPATH=$tree SGLANG_OMNI_CODEC_COMPILE_FORM=$form SGLANG_OMNI_CODEC_OBLIVIOUS=$oblivious \
      TORCHINDUCTOR_CACHE_DIR=$W/cache_$arm/inductor TRITON_CACHE_DIR=$W/cache_$arm/triton \
      timeout 1800 python3 $K/codec_forms.py --out $W/${arm}_$phase $extra > $W/${arm}_$phase.log 2>&1)
    local status=$?
    echo "$arm $phase exit $status wall $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T): $(tail -n 1 $W/${arm}_$phase.log | cut -c1-220)" >> $L
    if [ $status -ne 0 ]; then
      return 1
    else
      :
    fi
  done
}

run_arm ops $FORMS ops 0 ""
run_arm all_obl $FORMS all 1 ""
run_arm all $FORMS all 0 ""
run_arm batch $FORMS batch 0 ""
run_arm batch_obl $FORMS batch 1 ""
run_arm main $MAIN static 0 ""
run_arm main_noap $MAIN static 0 "--no-autotune-pointwise"
python3 $K/compare_forms.py $W main ops batch batch_obl all all_obl main_noap > $W/compare.txt 2>&1
echo "queue block43 done $(date +%T)" >> $L
