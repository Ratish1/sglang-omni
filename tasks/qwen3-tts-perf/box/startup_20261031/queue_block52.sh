#!/usr/bin/env bash
# Block 52 of 10-31: the codec graph runners' eager warm-ups per captured shape (main: 3) at 3, 2 and 1, on PR B's tree
# and its warm cache from block 45, card 0, one fresh process each, 3 twice for the band: vocoder build and capture
# seconds, compiled replay times and waveforms against the first 3 (one slot per row). Waits for block 51 first.
# usage: D=... T=<kit tree> HEAD=<tree> bash queue_block52.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
HEAD=${HEAD:?the PR B tree}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/block52
L=$W/queue.log
mkdir -p $W
until grep -q "queue block51 done" $D/block51/queue.log 2> /dev/null; do sleep 20; done
echo "tree $HEAD $(git -C $HEAD rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $HEAD diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block52 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
run() {
  local name=$1 warmups=$2
  (cd $HEAD && PYTHONPATH=$HEAD TORCHINDUCTOR_CACHE_DIR=$D/block45/cache_head/inductor TRITON_CACHE_DIR=$D/block45/cache_head/triton \
    timeout 1800 python3 $K/codec_forms.py --out $W/${name}_warm --warmups $warmups > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T)" >> $L
}
run three_a 3
run three_b 3
run two 2
run one 1
python3 $K/compare_forms.py $W three_a three_b two one > $W/compare.txt 2>&1
head -n 5 $W/compare.txt >> $L
echo "queue block52 done $(date +%T)" >> $L
