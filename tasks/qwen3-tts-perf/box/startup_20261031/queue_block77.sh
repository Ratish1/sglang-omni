#!/usr/bin/env bash
# Block 77 of 11-02: the shared programs' first traced shape. Inductor takes each symbolic program's kernels and tuned
# configs from the shape it first traces (block 76: main's own (8, 8) program has the launches the PR's symbolic program
# carries to small sizes). TRACE=<scratch tree> precompiles SGLANG_OMNI_CODEC_TRACE_FIRST before the first capture, so
# each size class is traced first at the named shape. Main, the PR and each trace order through codec_kernels.py
# --all-widths on card 0: cold into its own cache, then two warm runs.
# usage: D=... T=<kit tree> MAIN=<tree> PR=<tree> TRACE=<tree> bash queue_block77.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
PR=${PR:?the PR tree}
TRACE=${TRACE:?the scratch tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block77
L=$W/queue.log
mkdir -p $W
for arm in MAIN PR TRACE; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block77 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
run() {
  local tree=$1 cache=$2 out=$3 first=$4
  (cd $tree && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$tree SGLANG_OMNI_CODEC_TRACE_FIRST=$first TORCHINDUCTOR_CACHE_DIR=$W/cache_$cache/inductor TRITON_CACHE_DIR=$W/cache_$cache/triton \
    timeout 2400 python3 $K/startup_20261031/codec_kernels.py --all-widths --out $W/$out > $W/$out.log 2>&1)
  echo "kernels $out exit $? $(date +%T) build $(python3 -c "import json;print(json.load(open('$W/$out.json'))['build_s'])" 2>/dev/null)" >> $L
}
arm() {
  local tree=$1 name=$2 first=$3
  run $tree $name ${name}_cold "$first"
  run $tree $name ${name}_warm1 "$first"
  run $tree $name ${name}_warm2 "$first"
}
arm $MAIN main ""
arm $PR pr ""
arm $TRACE small 1x2,2x1,2x2,2x3,1x1
arm $TRACE mid 1x4,4x1,4x4,4x3,1x1
arm $TRACE width8 1x8,2x8,2x1,2x3,1x1
python3 $K/startup_20261031/trace_first_compare.py $W main=main_warm1,main_warm2 pr=pr_warm1,pr_warm2 small=small_warm1,small_warm2 mid=mid_warm1,mid_warm2 width8=width8_warm1,width8_warm2 > $W/compare.txt 2>&1
cat $W/compare.txt >> $L
echo "queue block77 done $(date +%T)" >> $L
