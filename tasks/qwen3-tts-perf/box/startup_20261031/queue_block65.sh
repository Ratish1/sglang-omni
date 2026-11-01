#!/usr/bin/env bash
# Block 65 of 11-01: #3390's codes against official qwen-tts's float32 exact-length encode were measured for one compile
# (0.9935, eager 0.9936); #3409 leaves one compile state. encoder_speedups_gate.py's served variant (the tree as the
# server loads it, its batcher graphs and its eager encode) on the 666 SeedTTS EN reference clips, main twice and #3409
# twice, each into a fresh Inductor cache, card 0, one at a time: code agreement with official for graphs and eager,
# device time and kernels per encode over 200 clips under the profiler.
# usage: D=... T=<kit tree> MAIN=<tree> FIX=<tree> WAVS=<dir> bash queue_block65.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
FIX=${FIX:?the fix tree}
WAVS=${WAVS:?the reference clip directory}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block65
L=$W/queue.log
mkdir -p $W
for arm in MAIN FIX; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block65 start $(date +%T) kit $(git -C $T rev-parse HEAD); clips $(ls $WAVS | wc -l); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
gate() {
  local name=$1 tree=$2
  (cd $tree && HF_HUB_OFFLINE=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_$name/inductor TRITON_CACHE_DIR=$W/cache_$name/triton \
    timeout 1800 python3 $K/next_20261029/encoder_speedups_gate.py $WAVS --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --card 0 \
    --clips 666 --variants served > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T): $(grep -E '^== served|official /|best_config' $W/$name.log | tr '\n' '#') best_config files $(find $W/cache_$name -name '*.best_config' | wc -l)" >> $L
}
gate main_a $MAIN
gate fix_a $FIX
gate fix_b $FIX
gate main_b $MAIN
echo "queue block65 done $(date +%T)" >> $L
