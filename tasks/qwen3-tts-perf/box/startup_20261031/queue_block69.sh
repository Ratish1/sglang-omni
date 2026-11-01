#!/usr/bin/env bash
# Block 69 of 11-01: block 68 showed compiled widths closer to float32 than eager ones and every width compiled 19 %
# faster on the follow-up runner, but per-shape or per-region programs make that cost startup. The whole step with
# batch and width symbolic (the size classes, 4 programs) covers every width without new traces. Scratch
# test/qwen3-tts-codec-size-class-every-width (the size-class commits on main, SGLANG_OMNI_CODEC_COMPILE_ALL_WIDTHS=1
# compiles every cold, window and warm width): codec_forms.py --all-keys --fp32-reference, size classes on main's widths
# and on every width, each cold into its own cache then warm, then main (block 67's warm cache) and both size-class arms
# warm again, card 0, one at a time. fp32_forms.py against main.
# usage: D=... T=<kit tree> MAIN=<tree> SIZES=<tree> bash queue_block69.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
SIZES=${SIZES:?the size-class tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/block69
L=$W/queue.log
mkdir -p $W
for arm in MAIN SIZES; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block69 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
run() {
  local name=$1 tree=$2 cache=$3 widths=$4
  (cd $tree && PYTHONPATH=$tree SGLANG_OMNI_CODEC_COMPILE_ALL_WIDTHS=$widths TORCHINDUCTOR_CACHE_DIR=$cache/inductor \
    TRITON_CACHE_DIR=$cache/triton timeout 2400 python3 $K/codec_forms.py --all-keys --fp32-reference --out $W/$name \
    > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T)" >> $L
}
run sizes_cold $SIZES $W/cache_sizes 0
run sizes_warm $SIZES $W/cache_sizes 0
run sizes_all_cold $SIZES $W/cache_sizes_all 1
run sizes_all_warm $SIZES $W/cache_sizes_all 1
run main_warm $MAIN $D/block67/cache_main 0
run sizes_all_warm2 $SIZES $W/cache_sizes_all 1
run sizes_warm2 $SIZES $W/cache_sizes 0
python3 - $W >> $L 2>&1 <<'EOF'
import json
import sys

directory = sys.argv[1]
for name in ("main_warm", "sizes_cold", "sizes_warm", "sizes_warm2", "sizes_all_cold", "sizes_all_warm", "sizes_all_warm2"):
    data = json.load(open(f"{directory}/{name}.json"))
    print(f"{name:<16} build {data['build_s']:6.1f} precompile {data['precompile_total_s']:6.1f} programs {data['unique_graphs']:3d} "
          f"captures {' '.join(str(round(item['seconds'], 1)) for item in data['captures'])}")
EOF
python3 $K/fp32_forms.py $W main_warm sizes_warm sizes_warm2 sizes_all_warm sizes_all_warm2 > $W/fp32.txt 2>&1
cat $W/fp32.txt >> $L
echo "queue block69 done $(date +%T)" >> $L
