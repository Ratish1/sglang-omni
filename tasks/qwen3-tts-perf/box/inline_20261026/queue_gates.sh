#!/usr/bin/env bash
# The real-checkpoint gates on main, the PR and inline, after queue_inline.sh ends (one process on the card): inline's
# run failed one vocoder decoder graph capture (test_pipeline.py, 1.7B, batch 3 at 256 frames, a Triton launch inside
# the capture), in code the PR does not touch. Each tree with the container's Triton cache, then with an empty one.
set -u
D=/data/inline_20261026
M=${M:?the main tree}
P=${P:?the PR tree}
I=${I:?the inline tree}
L=$D/queue.log
until grep -q queue_inline_done $L; do
  sleep 15
done
SNAP17=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
SNAP06=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/* | head -n 1)
echo "queue gates start $(date +%T)" >> $L
for arm in "main $M" "pr $P" "inline $I"; do
  set -- $arm
  for cache in container empty; do
    if [ $cache = empty ]; then
      CACHE_DIR=$D/triton_empty_$1
      mkdir -p $CACHE_DIR
    else
      CACHE_DIR=/root/.triton/cache
    fi
    (cd $2 && CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=$CACHE_DIR QWEN3_TTS_TOKENIZER_PATH=$SNAP17/speech_tokenizer \
      QWEN3_TTS_BASE_1_7B_PATH=$SNAP17 QWEN3_TTS_BASE_0_6B_PATH=$SNAP06 PYTHONPATH=$2 timeout 1800 python3 -m pytest \
      tests/unit_test/qwen3_tts -m benchmark -k "test_real_" -q -p no:cacheprovider -rfEs \
      > $D/units/gates_$1_$cache.log 2>&1)
    echo "gates $1 $cache triton cache rc $? $(grep -E 'passed|failed' $D/units/gates_$1_$cache.log | tail -n 1) $(grep -E '^FAILED' $D/units/gates_$1_$cache.log | sed -E 's/.*\[//; s/\]$//' | tr '\n' ' ')$(date +%T)" >> $L
  done
done
echo "queue_gates_done $(date +%T)" >> $L
