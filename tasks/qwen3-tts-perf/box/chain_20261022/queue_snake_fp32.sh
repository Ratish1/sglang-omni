#!/usr/bin/env bash
# #3097 with its fused SnakeBeta computed in fp32 and rounded once (perf/qwen3-tts-vocoder-snake-fp32, one commit on
# #3097's head), the fix for its compiled decode sitting 0.06 to 0.13 dB further from the fp32 decoder than main's.
# Units: qwen3_tts, the kernels and Qwen3-Omni's code2wav suites (the kernel file is shared; Qwen3-Omni keeps the eager
# rounding by default), the real-checkpoint gates; a failure runs nothing else. Then the truth probe (served runners,
# width 8 compiled) against main's run of the same probe, and vocoder_decode_bench.py #3097 against the fix in both
# card orders.
set -u
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
P=$HERE/vocoder_truth_probe.py
LIST=/data/prof_20261022/formal/meta_w2/long/L1_c16_run.lst
M=${M:?the main tree}
P2=${P2:?the 3097 tree}
PF=${PF:?the fix tree}
O=tests/unit_test/qwen3_omni
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
echo "queue snake fp32 start $(date +%T) d2 $(git -C $P2 rev-parse HEAD:sglang_omni) fix $(git -C $PF rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $PF && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$PF timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/kernels \
  $O/test_code2wav.py $O/test_code2wav_cuda_graph.py $O/test_code2wav_batching.py $O/test_code2wav_overlap.py \
  $O/test_code2wav_snake_beta.py -q -p no:cacheprovider -rfEs > $D/units/snake_fp32.log 2>&1)
UNITS=$?
echo "units snake fp32 rc $UNITS $(grep -E 'passed|failed' $D/units/snake_fp32.log | tail -n 1) $(date +%T)" >> $L
(cd $PF && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$PF timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py \
  tests/unit_test/qwen3_tts/test_pipeline.py -m benchmark -k "test_real_" -q -p no:cacheprovider -rfEs \
  > $D/units/gates_snake_fp32.log 2>&1)
GATES=$?
echo "gates snake fp32 rc $GATES $(grep -E 'passed|failed' $D/units/gates_snake_fp32.log | tail -n 1) $(date +%T)" >> $L
if [ $UNITS -ne 0 ] || [ $GATES -ne 0 ]; then
  echo "stopped: snake fp32 units or gates failed" >> $L
  echo "queue_snake_fp32_done $(date +%T)" >> $L
  exit 1
else
  :
fi
(cd $PF && TORCHINDUCTOR_CACHE_DIR=$D/inductor_truth_snake_fp32 PYTHONPATH=$PF timeout 1800 python3 $P run $LIST \
  $D/truth5_snake_fp32.json --card 3 --compiled > $D/truth5_snake_fp32.txt 2>&1)
(cd $M && PYTHONPATH=$M python3 $P compare $D/truth5_main_compiled.json $D/truth5_snake_fp32.json > $D/truth5_main_compiled_snake_fp32.txt 2>&1)
(cd $M && PYTHONPATH=$M python3 $P compare $D/truth5_d2_compiled.json $D/truth5_snake_fp32.json > $D/truth5_d2_compiled_snake_fp32.txt 2>&1)
echo "truth snake fp32 $(grep 'stream:' $D/truth5_main_compiled_snake_fp32.txt | cut -c1-160 | tr '\n' ' ') $(date +%T)" >> $L
for arm in "d2 $P2" "fix $PF" "fix $PF" "d2 $P2"; do
  set -- $arm
  echo "== $1 $(date +%T)" >> $D/bench_snake_fp32.log
  (cd $2 && TORCHINDUCTOR_CACHE_DIR=$D/inductor_bench_snake_$1 PYTHONPATH=$2 python3 $HERE/vocoder_decode_bench.py --card 3 2>&1 \
    | grep -E "^(cold|warm|tail) |Error" >> $D/bench_snake_fp32.log)
done
echo "queue_snake_fp32_done $(date +%T)" >> $L
