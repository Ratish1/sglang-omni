#!/usr/bin/env bash
# Block 12 of 10-29, after block 11 (WAIT_PID, its queue's pid). Card 0: the nsys thread probe (block 8's flags, then
# cuda and nvtx alone), each export read per thread; then the limit branch's qwen3_tts suite. Card 1: the encoder
# speedups gate on main's bf16 encoder, then on D's float32 encoder. The SeedTTS EN reference clips are staged first.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
REPO=/workspace/sglang-omni
K=$T/tasks/qwen3-tts-perf/box/next_20261029
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block12
L=$W/queue.log
mkdir -p $W/nsys_probe
if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 $WAIT_PID 2> /dev/null; do sleep 30; done
else
  :
fi
echo "queue block12 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
git -C $REPO fetch -q origin fix/qwen3-tts-task-checkpoint-limit
rm -rf /workspace/n29_limit
git -C $REPO worktree prune
git -C $REPO worktree add -q --detach /workspace/n29_limit 051b93a42
echo "tree limit $(git -C /workspace/n29_limit rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
if [ ! -d $D/wavs ]; then
  (cd /workspace/n29_main && PYTHONPATH=/workspace/n29_main python3 $T/tasks/qwen3-tts-perf/box/bisect_20261027/stage_seedtts_wavs.py $D/wavs >> $L 2>&1)
else
  :
fi

lane_zero() {
  for flags in full minimal; do
    if [ $flags = full ]; then trace=cuda,nvtx,osrt,python-gil; else trace=cuda,nvtx; fi
    (cd /workspace/n29_main && PYTHONPATH=/workspace/n29_main nsys profile -o $W/nsys_probe/$flags --force-overwrite=true \
      --trace=$trace --cuda-graph-trace=graph --sample=none --cpuctxsw=none \
      python3 $K/nsys_thread_probe.py run $D/wavs --model $M17 --card 0 > $W/nsys_probe/$flags.log 2>&1)
    echo "nsys probe $flags rc $? $(date +%T)" >> $L
    nsys export --type=sqlite --force-overwrite=true --output=$W/nsys_probe/$flags.sqlite $W/nsys_probe/$flags.nsys-rep >> $W/nsys_probe/$flags.log 2>&1
    python3 $K/nsys_thread_probe.py read $W/nsys_probe/$flags.sqlite >> $W/nsys_probe.txt 2>&1
    echo "nsys probe $flags read rc $? $(date +%T)" >> $L
  done
  (cd /workspace/n29_limit && CUDA_VISIBLE_DEVICES=0 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider > $W/units_limit.txt 2>&1)
  echo "== units limit rc $?: $(grep -E '[0-9]+ passed|failed|error' $W/units_limit.txt | tail -n 1) $(date +%T)" >> $L
}

lane_one() {
  for tree in main refenc; do
    (cd /workspace/n29_$tree && PYTHONPATH=/workspace/n29_$tree TORCHINDUCTOR_CACHE_DIR=$W/inductor_$tree \
      TORCH_LOGS=graph_breaks,recompiles \
      python3 $K/encoder_speedups_gate.py $D/wavs --model $M17 --card 1 > $W/gate_$tree.txt 2>&1)
    echo "gate $tree rc $? $(grep -E '^== ' $W/gate_$tree.txt | tr '\n' ';') $(date +%T)" >> $L
  done
}

lane_zero &
zero=$!
lane_one &
one=$!
wait $zero $one
echo "queue block12 done $(date +%T)" >> $L
