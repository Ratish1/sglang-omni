#!/usr/bin/env bash
# Block 22 of 10-29: startup of the compiled reference encode, static (one compile per bucket) against one dynamic
# compile, each lane in its own fresh inductor cache (cold), then again on that cache (warm); main tree, card 0 static,
# card 1 dynamic. Written into the container as block22/run.sh and run from there, copied here as it ran.
# usage: bash queue_block22.sh KIT_TREE RESULTS_DIR
set -u
T=$1; W=$2; K=$T/tasks/qwen3-tts-perf/box/next_20261029
lane() {
  local variant=$1 card=$2
  for pass in cold warm; do
    (cd /workspace/n29_main && PYTHONPATH=/workspace/n29_main TORCHINDUCTOR_CACHE_DIR=$W/inductor_$variant TORCH_LOGS=recompiles       python3 $K/encoder_speedups_gate.py /data/next_20261029/wavs --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --card $card --variants $variant > $W/gate_${variant}_$pass.txt 2>&1)
    echo "$variant $pass rc $? $(grep -E '^== |official|base /' $W/gate_${variant}_$pass.txt | tr '\n' ';') recompiles $(grep -c 'Recompiling function' $W/gate_${variant}_$pass.txt) $(date +%T)" >> $W/queue.log
  done
}
echo "block22 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $W/queue.log
lane compile 0 &
lane compile_dynamic 1 &
wait
echo "block22 done $(date +%T)" >> $W/queue.log
