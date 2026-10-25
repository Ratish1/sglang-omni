#!/usr/bin/env bash
# After queue_kvip_c128.sh: units and the real-checkpoint tests on the PR head after the ring clone was dropped
# (a95745edc; no caller clones an arena state).
set -u
D=/data/kvip1025
H=${H:?the PR head tree}
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
until grep -q queue_kvip_c128_done $L; do
  sleep 15
done
echo "queue kvip head2 start $(date +%T) head $(git -C $H rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
(cd $H && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$H timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts -q \
  -p no:cacheprovider -rfEs > $D/units/head2.log 2>&1)
echo "units head2 rc $? $(grep -E 'passed|failed' $D/units/head2.log | tail -n 1) $(date +%T)" >> $L
(cd $H && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$H timeout 900 python3 -m pytest tests/unit_test/qwen3_tts -m benchmark -k "test_real_" -q \
  -p no:cacheprovider -rfEs > $D/units/gates_head2.log 2>&1)
echo "gates head2 rc $? $(grep -E 'passed|failed' $D/units/gates_head2.log | tail -n 1) $(date +%T)" >> $L
echo "queue_kvip_head2_done $(date +%T)" >> $L
