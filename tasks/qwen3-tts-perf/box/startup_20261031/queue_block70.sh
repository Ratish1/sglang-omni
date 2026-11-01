#!/usr/bin/env bash
# Block 70 of 11-01: block 69's size classes on every width spent 3 extra programs and the wide window widths' gathered
# step (wider than the ring's in-place width, 8: streaming_vocoder.py:953-957), which compiled replays slower. Scratch
# test/qwen3-tts-codec-size-class-every-width with SGLANG_OMNI_CODEC_COMPILE_ALL_WIDTHS=inplace compiles every cold,
# window and warm width up to the in-place width and leaves 16, 32 and 64 eager. codec_forms.py --all-keys
# --fp32-reference cold into its own cache, then warm twice, card 0; fp32_forms.py against block 69's main and size-class
# arms.
# usage: D=... T=<kit tree> SIZES=<tree> bash queue_block70.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
SIZES=${SIZES:?the size-class tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/block70
L=$W/queue.log
mkdir -p $W
echo "tree SIZES $SIZES $(git -C $SIZES rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $SIZES diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block70 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
run() {
  local name=$1
  (cd $SIZES && PYTHONPATH=$SIZES SGLANG_OMNI_CODEC_COMPILE_ALL_WIDTHS=inplace TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor \
    TRITON_CACHE_DIR=$W/cache/triton timeout 2400 python3 $K/codec_forms.py --all-keys --fp32-reference --out $W/$name \
    > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T)" >> $L
}
run inplace_cold
run inplace_warm
run inplace_warm2
python3 - $W >> $L 2>&1 <<'EOF'
import json
import sys

directory = sys.argv[1]
for name in ("inplace_cold", "inplace_warm", "inplace_warm2"):
    data = json.load(open(f"{directory}/{name}.json"))
    print(f"{name:<14} build {data['build_s']:6.1f} precompile {data['precompile_total_s']:6.1f} programs {data['unique_graphs']:3d} "
          f"captures {' '.join(str(round(item['seconds'], 1)) for item in data['captures'])} compiled {data['compiled_shapes']}")
EOF
B=../block69
python3 $K/fp32_forms.py $W $B/main_warm $B/sizes_warm $B/sizes_all_warm inplace_warm inplace_warm2 > $W/fp32.txt 2>&1
cat $W/fp32.txt >> $L
echo "queue block70 done $(date +%T)" >> $L
