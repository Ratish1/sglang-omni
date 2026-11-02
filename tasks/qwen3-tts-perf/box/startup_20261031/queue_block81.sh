#!/usr/bin/env bash
# Block 81 of 11-02: saved programs on #3418's final head, on card 1. Its touched suites; codec_kernels.py cold into its own
# cache then warm, every graph's waveform against block 78's rows head (the same decode through an ordinary compile), and
# replay against it; boots of the saved tree, cold into its own cache then three warm, then #3418's final head three warm
# in its block 79 cache, alternating.
# usage: D=... T=<kit tree> SAVED=<tree> PR=<tree> bash queue_block81.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
SAVED=${SAVED:?the saved programs tree}
PR=${PR:?the PR tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block81
L=$W/queue.log
mkdir -p $W/logs
for arm in SAVED PR; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block81 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $SAVED && CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$W/suite_cache/inductor TRITON_CACHE_DIR=$W/suite_cache/triton timeout 3600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/kernels > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
grep -E "^FAILED|^ERROR" $W/suite.log | head -n 8 >> $L
run() {
  local out=$1
  (cd $SAVED && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$SAVED TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor TRITON_CACHE_DIR=$W/cache/triton \
    timeout 2400 python3 $K/startup_20261031/codec_kernels.py --all-widths --out $W/$out > $W/$out.log 2>&1)
  echo "kernels $out exit $? $(date +%T) $(python3 -c "import json;d=json.load(open('$W/$out.json'));print('build',d['build_s'],'programs',d['programs'],'keys',len(d['keys']))" 2>/dev/null)" >> $L
}
run saved_cold
run saved_warm
python3 - $W $D/block78 >> $L 2>&1 <<'EOF'
import json
import sys

import torch

directory, reference_directory = sys.argv[1:]
reference = torch.load(f"{reference_directory}/rows_all.pt")
saved = torch.load(f"{directory}/saved_warm.pt")
cold = torch.load(f"{directory}/saved_cold.pt")
print(f"waveforms bit-equal to the ordinary compile: warm {sum(torch.equal(reference[key], saved[key]) for key in reference)}, cold {sum(torch.equal(reference[key], cold[key]) for key in reference)} of {len(reference)}")
a = json.load(open(f"{reference_directory}/rows_all.json"))["keys"]
b = json.load(open(f"{directory}/saved_warm.json"))["keys"]
print(f"replay summed: ordinary compile {sum(a[key]['ms'] for key in a):.2f} ms, saved programs {sum(b[key]['ms'] for key in a):.2f} ms; launches at 2x2 {len(a['cold:2x2']['kernels'])} / {len(b['cold:2x2']['kernels'])}")
EOF
du -sh $W/cache/inductor/qwen3_tts_codec >> $L 2>&1

boot() {
  local label=$1 tree=$2 cache=$3 port=$4
  local began=$(date +%s.%N)
  (cd $tree && CUDA_VISIBLE_DEVICES=1 SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$cache/inductor TRITON_CACHE_DIR=$cache/triton \
    setsid bash -c "echo \$\$ > $W/logs/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port \
    --model-path $M17" > $W/logs/serve_$label.log 2>&1 < /dev/null &)
  sleep 1
  local pgid=$(cat $W/logs/serve_$label.pid)
  until kill -0 $pgid 2> /dev/null && curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null; then
      echo "$label died; $(grep -E 'Error|error' $W/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 0.5
    fi
  done
  echo "$label healthy after $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $began}") s $(date +%T); disabled $(grep -c 'capture disabled' $W/logs/serve_$label.log), unloaded $(grep -c 'Could not load\|traced from another' $W/logs/serve_$label.log)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
boot cold_saved $SAVED $W/boot_saved 25000
boot warm_pr_0 $PR $W/boot_pr 25200
boot warm_saved_1 $SAVED $W/boot_saved 25400
boot warm_pr_1 $PR $W/boot_pr 25600
boot warm_pr_2 $PR $W/boot_pr 25800
boot warm_saved_2 $SAVED $W/boot_saved 26000
boot warm_saved_3 $SAVED $W/boot_saved 26200
boot warm_pr_3 $PR $W/boot_pr 26400
python3 $K/startup_20261031/boot_split.py $W > $W/split.txt 2>&1
tail -n 4 $W/split.txt >> $L
for log in $W/logs/serve_warm_*_1.log; do echo "$(basename $log): $(grep -m1 -oE '#tokens: [0-9]+' $log)" >> $L; done
echo "queue block81 done $(date +%T)" >> $L
