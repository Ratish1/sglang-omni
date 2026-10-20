#!/usr/bin/env bash
# V5's seeded c1 pair against V3 was byte identical on 2 of 1088 requests: the vocoder's eager and compiled decodes per
# compiled shape on V3 and V5, compared across the trees. Then skinny GEMM candidates for the talker's and predictor's
# dense projections at 1 to 64 rows (cuBLAS, SGLang's Hopper GEMV, a Triton split-K GEMM), the core shared by items P7
# and T1. Runs alone on card 0 between queue_v5.sh and queue_v4.sh, so no pair shares the device.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
S=$T/tasks/omni_step_profiling_20260918/scripts
L=$D/queue.log
until grep -q queue_v5_done $L; do sleep 30; done
# note (ratish): v5_c16_a's arm a transcription found port 9400 held by a listener outside both containers.
(cd /workspace/v3_6f6039ba6 && PYTHONPATH=/workspace/v3_6f6039ba6 SGLANG_OMNI_STRICT_PORT=1 timeout 1800 python3 -m \
  benchmarks.eval.benchmark_tts_seedtts --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --transcribe-only --lang en --port 31400 \
  --output-dir $D/pairs/v5_c16_a/a/bench > $D/pairs/v5_c16_a/a/score--transcribe-only.log 2>&1)
echo "rescored v5_c16_a a rc $? $(date +%T)" >> $L
for TREE in /workspace/v3_6f6039ba6 /workspace/v5_a962e3db6; do
  (cd $TREE && PYTHONPATH=$TREE timeout 1200 python3 $S/codec_paths_dump.py $D/codec_paths_$(basename $TREE).pt \
    > $D/codec_paths_$(basename $TREE).txt 2>&1)
  echo "codec paths $(basename $TREE) rc $? $(date +%T)" >> $L
done
python3 $S/codec_paths_compare.py $D/codec_paths_v3_6f6039ba6.pt $D/codec_paths_v5_a962e3db6.pt > $D/codec_paths_v3_v5.txt 2>&1
echo "codec paths compare rc $? $(date +%T)" >> $L
(cd /workspace/m_bcd0d8c68 && TRITON_CACHE_DIR=$D/triton_skinny timeout 2400 python3 $S/skinny_gemm_bench.py > $D/skinny_gemm_bench.txt 2>&1)
echo "skinny gemm bench rc $? $(date +%T)" >> $L
echo "queue_bench2_done $(date +%T)" >> $L
