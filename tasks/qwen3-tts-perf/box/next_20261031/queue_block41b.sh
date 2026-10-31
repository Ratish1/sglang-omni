#!/usr/bin/env bash
# Block 41b of 10-31: why the compile PR's cold capture takes 94.6 s where the gate's CosyVoice3 form takes 15.1 s with
# the same program (block 41). The forms differ in the conv padding arithmetic: the PR pads the right edge by -L % stride,
# a modulo that nests once per strided conv on a symbolic length; the gate uses Mimi's ceil division. COMP is the PR head;
# CAND is COMP with only that arithmetic changed to Mimi's. Card 0 alone: each served cold (fresh inductor and Triton
# caches) with Dynamo's per-phase compile times, CAND warm; then the qwen3_tts suite on CAND.
# usage: D=... T=<kit tree> COMP=<tree> CAND=<tree> bash queue_block41b.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
COMP=${COMP:?the compile tree}
CAND=${CAND:?the candidate tree}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block41b
L=$W/queue.log
WAVS=$D/block41/wavs
mkdir -p $W/logs
for tree in $COMP $CAND; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni tests || echo DIRTY)" >> $L
done
echo "queue block41b start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

gate() {
  local name=$1 tree=$2 phase=$3 clips=$4
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_$name/inductor TRITON_CACHE_DIR=$W/cache_$name/triton \
    TORCH_LOGS=recompiles python3 $K/next_20261029/encoder_speedups_gate.py $WAVS --model $M17 --card 0 \
    --variants served --clips $clips > $W/${name}_$phase.txt 2>&1)
  echo "$name $phase: $(grep -E '^== served' $W/${name}_$phase.txt | cut -c1-170) | $(grep -E 'official / graphs' $W/${name}_$phase.txt | tr -s ' ') | recompiles $(grep -c '\[__recompiles\] Recompiling' $W/${name}_$phase.txt) $(date +%T)" >> $L
  grep -A40 -E '^-- served compile phases' $W/${name}_$phase.txt | grep -E '^ *[A-Za-z_.<>]+ +[0-9.]+' | sort -k2 -n -r | head -n 8 | sed "s/^/  $name $phase phase | /" >> $L
}

for name in comp cand; do
  rm -rf $W/cache_$name
done
gate comp $COMP cold 1088
gate cand $CAND cold 1088
gate cand $CAND warm 50

(cd $CAND && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$CAND timeout 3600 python3 -m pytest -q -p no:cacheprovider \
  tests/unit_test/qwen3_tts > $W/logs/suite.log 2>&1)
echo "suite on CAND: $(grep -E '[0-9]+ (passed|failed)' $W/logs/suite.log | tail -n 1) $(grep -E '^(FAILED|ERROR) ' $W/logs/suite.log | head -n 5 | tr '\n' ' ') $(date +%T)" >> $L
echo "queue block41b done $(date +%T)" >> $L
