#!/usr/bin/env bash
# The real-checkpoint gates the unit queues skip without their path variables: the tokenizer's incremental decode
# against the whole decode and across SnakeBeta fusion, and the 1.7B base's padded whole-utterance graph against eager,
# on main, V3 and V4, between V3's pairs and V4's.
set -u
D=/data/q3t_20261021
L=$D/queue.log
SNAP=/data/hf/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3
until grep -q queue_v3m2_done $L; do sleep 30; done
for TREE in /workspace/main_f31cf9e9a /workspace/v3pr_76b0b7aa2 /workspace/v4pr_71e2d7b01; do
  NAME=gates_$(basename $TREE)
  (cd $TREE && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
    PYTHONPATH=$TREE timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py \
    tests/unit_test/qwen3_tts/test_pipeline.py -k "test_real_" -q -p no:cacheprovider -rfEs > $D/units/$NAME.log 2>&1)
  echo "$NAME rc $? $(grep -E 'passed|failed' $D/units/$NAME.log | tail -n 1) $(date +%T)" >> $L
done
echo "queue_gates_done $(date +%T)" >> $L
