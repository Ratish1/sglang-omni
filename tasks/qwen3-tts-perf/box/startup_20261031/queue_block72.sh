#!/usr/bin/env bash
# Block 72 of 11-01: block 71's candidate spent a fifth program on widths divisible by 3: the dilated conv's
# .contiguous() in channels_last_conv1d (utils/channels_last_conv.py:96) copies only when the dilation tail is not
# zero, so Dynamo guards on the width's divisibility. The new head returns the codec's dilated conv rows as a view
# (contiguous=False, the codec's slot conv only; every other caller keeps the copy). Suites: qwen3_tts, the channels-last
# conv and single-output conv tests. codec_forms.py --all-keys --fp32-reference cold into its own cache then warm, against
# block 71's candidate key by key. Seeded c1 byte identity, block 71's head (card 0) against the new head (card 1). Boots:
# the new head cold into its own cache, then three warm.
# usage: D=... T=<kit tree> OLD=<tree> NEW=<tree> bash queue_block72.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
OLD=${OLD:?the block 71 candidate tree}
NEW=${NEW:?the new head tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block72
L=$W/queue.log
mkdir -p $W/logs
for arm in OLD NEW; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block72 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $NEW && CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts \
  tests/unit_test/utils/test_channels_last_conv.py tests/unit_test/kernels/test_single_output_conv.py > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
for phase in cold warm; do
  (cd $NEW && PYTHONPATH=$NEW TORCHINDUCTOR_CACHE_DIR=$W/forms_cache/inductor TRITON_CACHE_DIR=$W/forms_cache/triton \
    timeout 2400 python3 $K/startup_20261031/codec_forms.py --all-keys --fp32-reference --out $W/new_$phase > $W/new_$phase.log 2>&1)
  echo "forms $phase exit $? $(date +%T)" >> $L
done
python3 - $W >> $L 2>&1 <<'EOF'
import json
import sys

import torch

directory = sys.argv[1]
for name in ("new_cold", "new_warm"):
    data = json.load(open(f"{directory}/{name}.json"))
    print(f"{name} build {data['build_s']} precompile {data['precompile_total_s']} programs {data['unique_graphs']} "
          f"new programs at {[item['shape'] for item in data['precompiles'] if item['new_graphs']]}")
old, new = torch.load(f"{directory}/../block71/cand_warm.pt"), torch.load(f"{directory}/new_warm.pt")
print(f"keys bit-equal to the block 71 candidate: {sum(torch.equal(old[key], new[key]) for key in old)} of {len(old)}")
EOF
python3 $K/startup_20261031/fp32_forms.py $W ../block69/main_warm ../block71/cand_warm new_warm > $W/fp32.txt 2>&1
grep -E "^replay" $W/fp32.txt >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $OLD --branch $NEW --root $W/identity --port 20000 > $W/identity.txt 2>&1)
echo "identity: $(grep -E 'byte identical|error lines|Error' $W/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L

boot() {
  local label=$1 port=$2
  local began=$(date +%s.%N)
  (cd $NEW && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$NEW TORCHINDUCTOR_CACHE_DIR=$W/boot/inductor TRITON_CACHE_DIR=$W/boot/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port \
    --model-path $M17" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 1
  local pgid=$(cat $W/logs/serve_$label.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "$label died; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 1
    fi
  done
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
boot cold_new 24000
boot warm_new_1 24200
boot warm_new_2 24400
boot warm_new_3 24600
python3 $K/startup_20261031/boot_split.py $W > $W/split.txt 2>&1
tail -n 3 $W/split.txt >> $L
echo "queue block72 done $(date +%T)" >> $L
