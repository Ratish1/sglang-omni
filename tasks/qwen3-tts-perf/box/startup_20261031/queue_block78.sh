#!/usr/bin/env bash
# Block 78 of 11-02: the ring step's rows in one launch, against main, on the #3418 head. codec_kernels.py on card 0: main
# and the rows head each cold into their own cache, warm main, rows, rows, main over the keys main compiles (no other key
# runs in those processes), then one --all-widths run each for every key and the waveforms, the PR head's among them.
# usage: D=... T=<kit tree> MAIN=<tree> PR=<tree> ROWS=<tree> bash queue_block78.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
PR=${PR:?the PR tree}
ROWS=${ROWS:?the rows tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block78
L=$W/queue.log
mkdir -p $W
for arm in MAIN PR ROWS; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block78 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
run() {
  local arm=$1 out=$2 extra=$3
  local tree=${!arm}
  (cd $tree && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_${arm,,}/inductor TRITON_CACHE_DIR=$W/cache_${arm,,}/triton \
    timeout 2400 python3 $K/startup_20261031/codec_kernels.py $extra --out $W/$out > $W/$out.log 2>&1)
  echo "kernels $out exit $? $(date +%T) $(python3 -c "import json;d=json.load(open('$W/$out.json'));print('build',d['build_s'],'programs',d['programs'],'keys',len(d['keys']))" 2>/dev/null)" >> $L
}
run MAIN main_cold ""
run ROWS rows_cold ""
run MAIN main_warm1 ""
run ROWS rows_warm1 ""
run ROWS rows_warm2 ""
run MAIN main_warm2 ""
python3 $K/startup_20261031/kernels_compare.py $W main_warm1,main_warm2 rows_warm1,rows_warm2 > $W/compare.txt 2>&1
grep -E "^(cold|warm0|window):" $W/compare.txt >> $L
run PR pr_all --all-widths
run MAIN main_all --all-widths
run ROWS rows_all --all-widths
python3 - $W >> $L 2>&1 <<'EOF'
import json
import sys

import torch

directory = sys.argv[1]
keys = json.load(open(f"{directory}/main_all.json"))["keys"]
reference = torch.load(f"{directory}/pr_all.pt")
rows = torch.load(f"{directory}/rows_all.pt")
main = torch.load(f"{directory}/main_all.pt")
print(f"rows waveforms bit-equal to the PR head: {sum(torch.equal(reference[key], rows[key]) for key in reference)} of {len(reference)}")
print(f"main waveforms bit-equal to the rows head: {sum(torch.equal(main[key], rows[key]) for key in main)} of {len(main)}")
data = {name: json.load(open(f"{directory}/{name}_all.json"))["keys"] for name in ("main", "pr", "rows")}
us = lambda name, key: sum(item["us"] for item in data[name][key]["kernels"])
compiled_on_main = [key for key in keys if key.startswith("cold:") or key.split(":")[1].startswith("8x")]
newly = [key for key in keys if key not in compiled_on_main]
for group, names in (("main's compiled keys", compiled_on_main), ("keys main runs eager", newly)):
    print(f"{group}: kernel us main {sum(us('main', key) for key in names):.0f} PR {sum(us('pr', key) for key in names):.0f} rows {sum(us('rows', key) for key in names):.0f}; "
          f"replay ms main {sum(data['main'][key]['ms'] for key in names):.2f} PR {sum(data['pr'][key]['ms'] for key in names):.2f} rows {sum(data['rows'][key]['ms'] for key in names):.2f}")
EOF
echo "queue block78 done $(date +%T)" >> $L
