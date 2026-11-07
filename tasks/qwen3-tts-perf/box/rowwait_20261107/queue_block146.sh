#!/usr/bin/env bash
# Block 146 of 11-07: does the initial worker's 2 ms window cost a whole request anything, one H200.
# Arms: M main, H #3477's head, N the head with a whole request ending the wait. Per model one priming boot a tree,
# then M H N N H M, each boot serving non-streaming c1 seeded, c16 and c64 over all 1,088 SeedTTS EN prompts.
# usage: D=... T=<kit tree> M=<tree> H=<tree> N=<tree> CPUS=<card 0 cpus> [PORT=12100] bash queue_block146.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
CPUS=${CPUS:?card 0 cpus}
PORT=${PORT:-12100}
K=$T/tasks/qwen3-tts-perf/box/rowwait_20261107
L=$D/queue.log
export CARD_CPUS="0=$CPUS"
mkdir -p $D
echo $$ > $D/queue.pgid
for arm in M H N; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
echo "queue start $(date +%T) kit $(git -C $T rev-parse HEAD); gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $L
port=$PORT
boot() {
  local root=$1 label=$2 arm=$3 model=$4 cells=$5
  local tree=${!arm}
  ARM_ROOT=$root TORCHINDUCTOR_CACHE_DIR=$D/cache/$arm/inductor TRITON_CACHE_DIR=$D/cache/$arm/triton \
    timeout 3600 python3 $K/arm_cells.py --label $label --tree $tree --model $model --port $port --cells $cells \
    >> $D/arms.out 2>&1
  echo "$label exit $? $(date +%T) startup $(cat $root/$label/startup.txt 2>/dev/null) $(tail -1 $D/arms.out | cut -c1-160)" >> $L
  port=$((port + 200))
}
for model in 06:Qwen/Qwen3-TTS-12Hz-0.6B-Base 17:Qwen/Qwen3-TTS-12Hz-1.7B-Base; do
  name=${model%%:*}
  path=${model#*:}
  for arm in M H N; do
    boot $D/prime ${name}_${arm}_0 $arm $path n64:64
  done
  for arm in M H N N H M; do
    count=$(ls -d $D/arms/${name}_${arm}_* 2>/dev/null | wc -l)
    boot $D/arms ${name}_${arm}_$((count + 1)) $arm $path n1:1:1234,n16:16,n64:64
  done
done
echo "queue done $(date +%T)" >> $L
python3 $K/rowwait_read.py --root $D/arms > $D/read.txt 2>&1
echo "read rc $? $(date +%T)" >> $L
