#!/usr/bin/env bash
# Block 133 of 11-05: the sequence behind the illegal address, on the commit that faulted, with a row left on the stateful path after teardown.
# Each variant in its own process, removing one step at a time; a variant that faults runs again with CUDA_LAUNCH_BLOCKING=1 to name the op.
# usage: D=... T=<kit tree> ARM=<tree> [BLOCK=block133] bash queue_block133.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
ARM=${ARM:?the tree}
BLOCK=${BLOCK:-block133}
S=$T/tasks/qwen3-tts-perf/box/startup_20261031
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD) arm $(git -C $ARM rev-parse HEAD HEAD:sglang_omni | tr '\n' ' '); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TRITON_CACHE_DIR=$D/cache/triton
variant() {
  local card=$1 name=$2
  shift 2
  (cd $ARM && PYTHONPATH=$ARM CUDA_VISIBLE_DEVICES=$card timeout 1200 python3 $S/whole_fault_read.py "$@" > $W/$name.log 2>&1)
  local rc=$?
  local faults=$(grep -ac '"completed": false' $W/$name.log)
  echo "$name [$*] rc $rc $(date +%T) completed $(grep -ac '"completed": true' $W/$name.log) failed $faults illegal $(grep -ac 'illegal memory access' $W/$name.log)" >> $L
  grep -a '"completed": false' $W/$name.log | cut -c1-260 >> $L
  if [ "$rc" != "0" ] || [ "$faults" != "0" ]; then
    (cd $ARM && PYTHONPATH=$ARM CUDA_VISIBLE_DEVICES=$card CUDA_LAUNCH_BLOCKING=1 timeout 1200 python3 $S/whole_fault_read.py "$@" > $W/${name}_blocking.log 2>&1)
    echo "$name blocking rc $? completed $(grep -ac '"completed": true' $W/${name}_blocking.log) failed $(grep -ac '"completed": false' $W/${name}_blocking.log)" >> $L
    grep -an -B28 -m1 'illegal memory access' $W/${name}_blocking.log | grep -aE 'File "|Error' | cut -c1-190 | tail -n 16 >> $L
  else
    :
  fi
}
(
  variant 0 full
  variant 0 only3 --only-case 3
) &
(
  variant 1 only3_noclean --only-case 3 --no-clean
  variant 1 only3_nonative --only-case 3 --no-native
  variant 1 only3_noclean_nonative --only-case 3 --no-clean --no-native
) &
wait
echo "queue $BLOCK done $(date +%T)" >> $L
