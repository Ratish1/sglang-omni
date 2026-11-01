#!/usr/bin/env bash
# Block 61b of 11-01: block 61's encoder probes died on a device-side assert in the probe's own replay timing (a
# bucket's graph replayed with the previous clip's longer clip_samples, past its tail padding index), on main and the
# fix alike; the probe now times each bucket at its full length. The encoder alone on the fix into three fresh caches
# over both cards and on main into one: codes compared (with block 59e's no-split run and eager) and graph replay
# device time per bucket. Waits for block 61.
# usage: D=... T=<kit tree> MAIN=<tree> FIX=<tree> bash queue_block61b.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
FIX=${FIX:?the fix tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block61b
L=$W/queue.log
mkdir -p $W
until grep -q "queue block61 done" $D/block61/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN FIX; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block61b start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
probe() {
  local name=$1 tree=$2 gpu=$3 cache=$4
  (cd $tree && HF_HUB_OFFLINE=1 PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/$cache/inductor TRITON_CACHE_DIR=$W/$cache/triton \
    timeout 1200 python3 $K/next_20261031/refenc_codes.py --gpu $gpu --out $W/$name > $W/$name.log 2>&1)
  echo "$name exit $? $(date +%T): $(tail -n 1 $W/$name.log)" >> $L
}
probe fix_card0_a $FIX 0 cacheFa
probe fix_card0_b $FIX 0 cacheFb
probe fix_card1_c $FIX 1 cacheFc
probe main_card0_a $MAIN 0 cacheMa
(cd $FIX && HF_HUB_OFFLINE=1 PYTHONPATH=$FIX python3 $K/next_20261031/refenc_codes.py --compare $W/fix_card0_a.pt \
  $W/fix_card0_b.pt $W/fix_card1_c.pt $W/main_card0_a.pt $D/block59e/nosplit_card0_a.pt $D/block59b/card0_eager.pt \
  > $W/compare.txt 2>&1)
grep -vE "INFO|WARN|Using the latest|Found the latest" $W/compare.txt >> $L
echo "queue block61b done $(date +%T)" >> $L
