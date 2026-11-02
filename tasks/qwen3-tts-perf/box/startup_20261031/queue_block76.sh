#!/usr/bin/env bash
# Block 76 of 11-02: the regression #3418 carries, kernel by kernel. Main and the PR head through codec_kernels.py on card
# 0, each cold into its own cache, then warm main, PR, PR, main; the graphs main already compiles (cold, warm0 and window
# width 8) compared per kernel.
# usage: D=... T=<kit tree> MAIN=<tree> PR=<tree> bash queue_block76.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
PR=${PR:?the PR tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block76
L=$W/queue.log
mkdir -p $W
for arm in MAIN PR; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block76 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
run() {
  local arm=$1 out=$2
  local tree=${!arm}
  (cd $tree && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_${arm,,}/inductor TRITON_CACHE_DIR=$W/cache_${arm,,}/triton \
    timeout 2400 python3 $K/startup_20261031/codec_kernels.py --out $W/$out > $W/$out.log 2>&1)
  echo "kernels $out exit $? $(date +%T)" >> $L
}
run MAIN main_cold
run PR pr_cold
run MAIN main_warm1
run PR pr_warm1
run PR pr_warm2
run MAIN main_warm2
python3 $K/startup_20261031/kernels_compare.py $W main_warm1,main_warm2 pr_warm1,pr_warm2 > $W/compare.txt 2>&1
grep -E "^(cold|warm0|window):" $W/compare.txt >> $L
echo "queue block76 done $(date +%T)" >> $L
