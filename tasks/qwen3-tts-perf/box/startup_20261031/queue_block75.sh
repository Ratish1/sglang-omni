#!/usr/bin/env bash
# Block 75 of 11-01: the #3418 review. NEW drops the single-output conv op (the final conv is a direct Triton launch in
# the trace again) and extends the size-class test to replay compiled graphs against fp32 and eager decodes; OPAQUE is
# the opened head. The qwen3_tts suite and the single-output conv tests on NEW; the test's fp32 bound over seeds 28 to
# 39 and the review's zeroed-final-conv mutation; codec_forms.py NEW cold into its own cache, then warm OPAQUE (block
# 74's cache), NEW, NEW, OPAQUE, key by key against OPAQUE.
# usage: D=... T=<kit tree> OPAQUE=<tree> NEW=<tree> bash queue_block75.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
OPAQUE=${OPAQUE:?the opened head tree}
NEW=${NEW:?the new head tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block75
L=$W/queue.log
mkdir -p $W
for arm in OPAQUE NEW; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block75 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $NEW && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 3600 \
  python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/kernels/test_single_output_conv.py > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
grep -E "^FAILED|Error" $W/suite.log | head -n 5 >> $L
(cd $NEW && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$W/seeds_cache/inductor TRITON_CACHE_DIR=$W/seeds_cache/triton timeout 3600 \
  python3 $K/startup_20261031/size_class_test_seeds.py $NEW 28 12 > $W/seeds.log 2>&1)
grep -E "^seed|^seeds|^mutation|Traceback|Error" $W/seeds.log >> $L
forms() {
  local arm=$1 cache=$2 out=$3
  local tree=${!arm}
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    timeout 2400 python3 $K/startup_20261031/codec_forms.py --all-keys --fp32-reference --out $W/$out > $W/$out.log 2>&1)
  echo "forms $out exit $? $(date +%T)" >> $L
}
forms NEW $W/cache_new new_cold
forms OPAQUE $D/block74/cache_pr opaque_warm1
forms NEW $W/cache_new new_warm1
forms NEW $W/cache_new new_warm2
forms OPAQUE $D/block74/cache_pr opaque_warm2
python3 $K/startup_20261031/fp32_forms.py $W opaque_warm1 opaque_warm2 new_warm1 new_warm2 > $W/fp32.txt 2>&1
grep -E "^replay" $W/fp32.txt >> $L
python3 - $W >> $L 2>&1 <<'EOF'
import json
import sys

import torch

directory = sys.argv[1]
for name in ("new_cold", "opaque_warm1", "new_warm1", "new_warm2", "opaque_warm2"):
    data = json.load(open(f"{directory}/{name}.json"))
    print(f"{name} build {data['build_s']} precompile {data['precompile_total_s']} programs {data['unique_graphs']} "
          f"new programs at {[item['shape'] for item in data['precompiles'] if item['new_graphs']]}")
opaque = torch.load(f"{directory}/opaque_warm1.pt")
for name in ("new_cold", "new_warm1", "new_warm2", "opaque_warm2"):
    other = torch.load(f"{directory}/{name}.pt")
    print(f"{name} keys bit-equal to opaque_warm1: {sum(torch.equal(opaque[key], other[key]) for key in opaque)} of {len(opaque)}")
EOF
echo "queue block75 done $(date +%T)" >> $L
