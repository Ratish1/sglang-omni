#!/usr/bin/env bash
# Block 39 of 10-31: the reference encode's compile forms, truly cold and then warm, one card, nothing else on the box.
# Block 38 measured the one-program form's capture at 95.6 s from a fresh inductor cache, against block 30's 16.9 s for
# the same form; block 30 reset TORCHINDUCTOR_CACHE_DIR per run but not Triton's own kernel cache, and block 38 ran
# beside another session's server boot. Here every cold run starts a fresh process with a fresh TORCHINDUCTOR_CACHE_DIR
# and a fresh TRITON_CACHE_DIR, and its warm run reuses both; encoder_speedups_gate.py prints capture time and Dynamo's
# per-phase compile times. Forms: per-bucket static (compile), the regional dynamic=True form (compile_regional), and
# CosyVoice3's one-program form as the gate builds it (compile_cosy), all on #3368's tree; and the compile PR's own
# one-program form (served, on its tree). Cold runs encode all 1,088 references (codes against official); warm runs
# only time the capture (50 clips).
# usage: D=... T=... DTREE=<tree> CTREE=<tree> CARD=<card> bash queue_block39.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
DTREE=${DTREE:?the #3368 tree}
CTREE=${CTREE:?the compile tree}
CARD=${CARD:?the card}
K=$T/tasks/qwen3-tts-perf/box
M17=Qwen/Qwen3-TTS-12Hz-1.7B-Base
W=$D/block39
L=$W/queue.log
mkdir -p $W
WAVS=$D/block38/wavs
echo "queue block39 start $(date +%T) refenc $(git -C $DTREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')compile $(git -C $CTREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) card $CARD; gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L

gate() {
  local name=$1 tree=$2 variant=$3 phase=$4 clips=$5
  (cd $tree && PYTHONPATH=$tree TORCHINDUCTOR_CACHE_DIR=$W/cache_$name/inductor TRITON_CACHE_DIR=$W/cache_$name/triton \
    TORCH_LOGS=recompiles python3 $K/next_20261029/encoder_speedups_gate.py $WAVS --model $M17 --card $CARD \
    --variants $variant --clips $clips > $W/${name}_$phase.txt 2>&1)
  echo "$name $phase: $(grep -E "^== $variant" $W/${name}_$phase.txt | cut -c1-170) | $(grep -E 'official / graphs' $W/${name}_$phase.txt | tr -s ' ') | recompiles $(grep -c '\[__recompiles\] Recompiling' $W/${name}_$phase.txt) $(date +%T)" >> $L
}

for spec in "static $DTREE compile" "regional $DTREE compile_regional" "cosy $DTREE compile_cosy" "pr $CTREE served"; do
  set -- $spec
  rm -rf $W/cache_$1
  gate $1 $2 $3 cold 1088
  gate $1 $2 $3 warm 50
done
echo "queue block39 done $(date +%T)" >> $L
