#!/usr/bin/env bash
# Block 77b of 11-02: block 77's arms that first traced a batch-1 shape hit Dynamo's recompile limit (8 programs, captures
# disabled), so they measured nothing; main and the PR there stand. Here each order first traces a shape above batch 1:
# 2x2, 2x8 and 4x4, then the other size classes. Each arm cold into its own cache, then two warm runs, compared with block
# 77's main and PR runs copied into this directory.
# usage: D=... T=<kit tree> TRACE=<tree> bash queue_block77b.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
TRACE=${TRACE:?the scratch tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block77b
L=$W/queue.log
mkdir -p $W
for run in main_warm1 main_warm2 pr_warm1 pr_warm2; do
  cp $D/block77/$run.json $D/block77/$run.pt $W/
done
echo "tree TRACE $TRACE $(git -C $TRACE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $TRACE diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block77b start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
run() {
  local cache=$1 out=$2 first=$3
  (cd $TRACE && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$TRACE SGLANG_OMNI_CODEC_TRACE_FIRST=$first TORCHINDUCTOR_CACHE_DIR=$W/cache_$cache/inductor TRITON_CACHE_DIR=$W/cache_$cache/triton \
    timeout 2400 python3 $K/startup_20261031/codec_kernels.py --all-widths --out $W/$out > $W/$out.log 2>&1)
  echo "kernels $out exit $? $(date +%T) $(python3 -c "import json;d=json.load(open('$W/$out.json'));print('build',d['build_s'],'programs',d['programs'],'keys',len(d['keys']))" 2>/dev/null) $(grep -c 'capture disabled' $W/$out.log) disabled" >> $L
}
arm() {
  local name=$1 first=$2
  run $name ${name}_cold "$first"
  run $name ${name}_warm1 "$first"
  run $name ${name}_warm2 "$first"
}
arm t2x2 2x2,2x3,2x1,1x2,1x1
arm t2x8 2x8,2x3,2x1,1x8,1x1
arm t4x4 4x4,4x3,4x1,1x4,1x1
python3 $K/startup_20261031/trace_first_compare.py $W main=main_warm1,main_warm2 pr=pr_warm1,pr_warm2 t2x2=t2x2_warm1,t2x2_warm2 t2x8=t2x8_warm1,t2x8_warm2 t4x4=t4x4_warm1,t4x4_warm2 > $W/compare.txt 2>&1
cat $W/compare.txt >> $L
echo "queue block77b done $(date +%T)" >> $L
