#!/usr/bin/env bash
# Block 74 of 11-01: main and the codec PR head through codec_forms.py --all-keys --fp32-reference in one run on one
# card, each cold into its own empty cache, then warm main, PR, PR, main. Build, precompile and programs per run; the
# float32 distance per runner; replay sums; per key, the PR against main in both warm pairs.
# usage: D=... T=<kit tree> MAIN=<tree> PR=<tree> bash queue_block74.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
PR=${PR:?the PR tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block74
L=$W/queue.log
mkdir -p $W
for arm in MAIN PR; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block74 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
forms() {
  local arm=$1 out=$2
  local tree=${!arm}
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_${arm,,}/inductor TRITON_CACHE_DIR=$W/cache_${arm,,}/triton \
    timeout 2400 python3 $K/startup_20261031/codec_forms.py --all-keys --fp32-reference --out $W/$out > $W/$out.log 2>&1)
  echo "forms $out exit $? $(date +%T)" >> $L
}
forms MAIN main_cold
forms PR pr_cold
forms MAIN main_warm1
forms PR pr_warm1
forms PR pr_warm2
forms MAIN main_warm2
python3 $K/startup_20261031/fp32_forms.py $W main_warm1 main_warm2 pr_warm1 pr_warm2 > $W/fp32.txt 2>&1
cat $W/fp32.txt >> $L
python3 - $W >> $L 2>&1 <<'EOF'
import json
import sys

directory = sys.argv[1]
runs = {}
for name in ("main_cold", "pr_cold", "main_warm1", "pr_warm1", "pr_warm2", "main_warm2"):
    data = json.load(open(f"{directory}/{name}.json"))
    runs[name] = data
    print(f"{name} build {data['build_s']} precompile {data['precompile_total_s']} programs {data['unique_graphs']}")


def times(name):
    return {(item["runner"], item["frames"], item["batch"]): item["ms"] for item in runs[name]["replays"]}


pairs = [(times("main_warm1"), times("pr_warm1")), (times("main_warm2"), times("pr_warm2"))]
keys = sorted(pairs[0][0])
ratios = {key: [pr[key] / main[key] - 1 for main, pr in pairs] for key in keys}
slower = [key for key in keys if all(value > 0.02 for value in ratios[key])]
faster = [key for key in keys if all(value < -0.02 for value in ratios[key])]
slower_any = [key for key in keys if all(value > 0 for value in ratios[key])]
print(f"keys {len(keys)}: PR more than 2 % slower in both pairs {len(slower)} {slower}; more than 2 % faster in both {len(faster)}")
print("PR slower in both pairs: " + "; ".join(
    f"{key} {pairs[0][0][key]:.4f}/{pairs[1][0][key]:.4f} -> {pairs[0][1][key]:.4f}/{pairs[1][1][key]:.4f} ms "
    f"({100 * ratios[key][0]:+.1f}, {100 * ratios[key][1]:+.1f} %)" for key in slower_any))
EOF
echo "queue block74 done $(date +%T)" >> $L
