#!/usr/bin/env bash
# Block 26 of 10-29: the reference encode compiled the SGLang default-path way (compile_regional: dynamic=True on the
# repeated regions only) against the static per-bucket compile, main tree, at the same time as block 22's lanes ran:
# card 0 regional, card 1 static, each in its own fresh inductor cache (cold), then again on that cache (warm). Each
# lane runs base first, so its codes compare against main's own graph codes.
# usage: WAIT_PID=PID bash queue_block26.sh KIT_TREE RESULTS_DIR
set -u
T=$1; W=$2; K=$T/tasks/qwen3-tts-perf/box/next_20261029
mkdir -p $W
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
lane() {
  local variant=$1 card=$2
  for pass in cold warm; do
    (cd /workspace/n29_main && PYTHONPATH=/workspace/n29_main TORCHINDUCTOR_CACHE_DIR=$W/inductor_$variant TORCH_LOGS=recompiles \
      python3 $K/encoder_speedups_gate.py /data/next_20261029/wavs --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --card $card \
      --variants base,$variant > $W/gate_${variant}_$pass.txt 2>&1)
    echo "$variant $pass rc $? $(grep -E '^== |official / graphs|base /' $W/gate_${variant}_$pass.txt | tr '\n' ';') recompiles $(grep -c 'Recompiling function' $W/gate_${variant}_$pass.txt) $(date +%T)" >> $W/queue.log
  done
}
echo "block26 start $(date +%T) main $(git -C /workspace/n29_main rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $W/queue.log
lane compile_regional 0 &
lane compile 1 &
wait
echo "block26 done $(date +%T)" >> $W/queue.log
