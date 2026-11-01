#!/usr/bin/env bash
# Block 67 of 11-01: the codec's compile in SGLang's regional form (multimodal_gen's regional_compile: repeated
# submodules compiled, fullgraph, dynamic=None) against main's whole step compiled per shape. Scratch branch
# test/qwen3-tts-codec-regional-compile: SGLANG_OMNI_CODEC_COMPILE_FORM=regions compiles frame_tensors, slot_upsample
# and slot_decoder_block for the decodes the graph runners ask compiled, the rest of the step eager;
# SGLANG_OMNI_CODEC_REGION_DYNAMIC=true makes them dynamic=True; SGLANG_OMNI_CODEC_COMPILE_ALL_WIDTHS=1 compiles every
# cold, window and warm width instead of the cold widths and the steady stride. codec_forms.py --all-keys builds the
# vocoder as serving does, each arm cold into its own cache then warm, card 0, one at a time: build and capture seconds,
# Dynamo programs, every key's replay device time and waveform against main's.
# usage: D=... T=<kit tree> MAIN=<tree> REGIONS=<tree> bash queue_block67.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
REGIONS=${REGIONS:?the regions tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/block67
L=$W/queue.log
mkdir -p $W
for arm in MAIN REGIONS; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block67 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
run() {
  local name=$1 tree=$2 phase=$3 form=$4 dynamic=$5 widths=$6
  (cd $tree && PYTHONPATH=$tree SGLANG_OMNI_CODEC_COMPILE_FORM=$form SGLANG_OMNI_CODEC_REGION_DYNAMIC=$dynamic \
    SGLANG_OMNI_CODEC_COMPILE_ALL_WIDTHS=$widths TORCHINDUCTOR_CACHE_DIR=$W/cache_$name/inductor \
    TRITON_CACHE_DIR=$W/cache_$name/triton timeout 2400 python3 $K/codec_forms.py --all-keys --out $W/${name}_$phase \
    > $W/${name}_$phase.log 2>&1)
  echo "$name $phase exit $? $(date +%T): $(tail -n 1 $W/${name}_$phase.log | cut -c1-200)" >> $L
}
for arm in "main $MAIN step none 0" "regions $REGIONS regions none 0" "regions_dyn $REGIONS regions true 0" \
    "regions_all $REGIONS regions none 1" "regions_dyn_all $REGIONS regions true 1"; do
  set -- $arm
  run $1 $2 cold $3 $4 $5
  run $1 $2 warm $3 $4 $5
done
python3 $K/compare_forms.py $W main regions regions_dyn regions_all regions_dyn_all > $W/compare.txt 2>&1
head -n 6 $W/compare.txt >> $L
echo "queue block67 done $(date +%T)" >> $L
