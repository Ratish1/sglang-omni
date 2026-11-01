#!/usr/bin/env bash
# Block 68 of 11-01: block 67's every-width arms differ from main by up to 0.23 relative L2 on the widths main runs
# eager and they compile, while compiled widths agree to 1e-3; the judge is the float32 whole decode of the same codes.
# codec_forms.py --all-keys --fp32-reference on block 67's warm caches, card 0, one at a time: main, regions every
# width, main with compile off (every key eager), regions, regions every width again, main again. fp32_forms.py:
# each arm's distance to float32 per runner, and replay sums for the process spread.
# usage: D=... T=<kit tree> MAIN=<tree> REGIONS=<tree> bash queue_block68.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
REGIONS=${REGIONS:?the regions tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
C=$D/block67
W=$D/block68
L=$W/queue.log
mkdir -p $W
for arm in MAIN REGIONS; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block68 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
run() {
  local name=$1 tree=$2 cache=$3 form=$4 widths=$5 flags=$6
  (cd $tree && PYTHONPATH=$tree SGLANG_OMNI_CODEC_COMPILE_FORM=$form SGLANG_OMNI_CODEC_REGION_DYNAMIC=none \
    SGLANG_OMNI_CODEC_COMPILE_ALL_WIDTHS=$widths TORCHINDUCTOR_CACHE_DIR=$C/cache_$cache/inductor \
    TRITON_CACHE_DIR=$C/cache_$cache/triton timeout 2400 python3 $K/codec_forms.py --all-keys --fp32-reference $flags \
    --out $W/$name > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T)" >> $L
}
run main_a $MAIN main step 0 ""
run all_a $REGIONS regions_all regions 1 ""
run eager $MAIN main step 0 --no-compile
run regions $REGIONS regions regions 0 ""
run all_b $REGIONS regions_all regions 1 ""
run main_b $MAIN main step 0 ""
python3 $K/fp32_forms.py $W eager main_a main_b regions all_a all_b > $W/fp32.txt 2>&1
cat $W/fp32.txt >> $L
echo "queue block68 done $(date +%T)" >> $L
