#!/usr/bin/env bash
# Block 30 of 10-29: the reference encode compiled in Fun-CosyVoice3's form (encoder_speedups_gate.py compile_cosy,
# compile_static_cosy, compile_regional_cosy) against block 26's two forms (compile per bucket, compile_regional), on
# main's bf16 encoder (card 0) and D's float32 encoder (card 1, fix/qwen3-tts-reference-encoder-float32-exact-length
# 19f0f8de5) at once. Each variant after base, in its own fresh inductor cache (cold), then again on it (warm), with
# Dynamo's per-phase compile times.
# usage: WAIT_PID=PID bash queue_block30.sh KIT_TREE RESULTS_DIR
set -u
T=$1; W=$2; K=$T/tasks/qwen3-tts-perf/box/next_20261029
mkdir -p $W
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
lane() {
  local tree=$1 card=$2 label=$3
  for variant in compile compile_static_cosy compile_cosy compile_regional compile_regional_cosy; do
    for pass in cold warm; do
      (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/inductor_${label}_$variant TORCH_LOGS=recompiles \
        python3 $K/encoder_speedups_gate.py /data/next_20261029/wavs --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --card $card \
        --variants base,$variant > $W/gate_${label}_${variant}_$pass.txt 2>&1)
      echo "$label $variant $pass rc $? $(grep -E '^== |official / graphs|base /' $W/gate_${label}_${variant}_$pass.txt | tr '\n' ';') recompiles $(grep -c 'Recompiling function' $W/gate_${label}_${variant}_$pass.txt) $(date +%T)" >> $W/queue.log
    done
  done
}
echo "block30 start $(date +%T) main $(git -C /workspace/n29_main rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')D $(git -C /workspace/n29_refenc rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $W/queue.log
lane /workspace/n29_main 0 main &
lane /workspace/n29_refenc 1 refenc &
wait
echo "block30 done $(date +%T)" >> $W/queue.log
