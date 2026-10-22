#!/usr/bin/env bash
# V3 rebased onto upstream main 3d7d2c306 (which carries P6, merged as dc8a48e11), cards 2 and 3 (0 and 1 are the
# CosyVoice container's): V3's touched unit suites and the real-checkpoint gates on both trees, then V3 against main as
# an unseeded c1 pair and c16 and c64 pairs in both card orders with SM Issue, full corpus, WER and SIM, then the
# readers (table, per-prompt pairing, first-audio tail).
set -u
export PAIR_ROOT=/data/v3_landing_20261022/pairs
D=/data/v3_landing_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
M=/workspace/main_3d7d2c306
V3=/workspace/v3pr_4d26f2092
O=tests/unit_test/qwen3_omni
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
mkdir -p $D/units $PAIR_ROOT
echo "queue v3 landing start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')v3 $(git -C $V3 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')t_marks $(git -C $T rev-parse HEAD)" >> $L
# the suites' GPU tests use cards 0 and 1 by name, and those are another container's; none of these suites reads
# the variable (only tests/unit_test/pipeline's wiring tests do)
units() {
  (cd $2 && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$2 timeout 3600 python3 -m pytest $3 -q -p no:cacheprovider -rfEs \
    > $D/units/$1.log 2>&1)
  local rc=$?
  echo "units $1 rc $rc tree $(git -C $2 rev-parse HEAD:sglang_omni) $(grep -E 'passed|failed' $D/units/$1.log | tail -n 1) $(date +%T)" >> $L
  return $rc
}
gates() {
  (cd $2 && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
    PYTHONPATH=$2 timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py \
    tests/unit_test/qwen3_tts/test_pipeline.py -k "test_real_" -q -p no:cacheprovider -rfEs > $D/units/gates_$1.log 2>&1)
  echo "gates $1 rc $? $(grep -E 'passed|failed' $D/units/gates_$1.log | tail -n 1) $(date +%T)" >> $L
}
run_pair() {
  python3 $IN/pair.py --label $1 --tree $M --b-tree $V3 --concurrency $2 --a-card $3 --b-card $4 --port $5 ${6:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)" >> $L
}
units v3m4 $V3 "tests/unit_test/qwen3_tts $O/test_code2wav.py $O/test_code2wav_cuda_graph.py $O/test_code2wav_batching.py $O/test_code2wav_overlap.py $O/test_code2wav_snake_beta.py"
V3_UNITS=$?
gates main_3d7d2c306 $M
gates v3pr_4d26f2092 $V3
if [ $V3_UNITS -eq 0 ]; then
  run_pair v3m4_c1 1 2 3 7000
  run_pair v3m4_c16_a 16 2 3 13000 --gpu-metrics
  run_pair v3m4_c16_b 16 3 2 18500 --gpu-metrics
  run_pair v3m4_c64_a 64 2 3 22000 --gpu-metrics
  run_pair v3m4_c64_b 64 3 2 26000 --gpu-metrics
  labels=$(ls $PAIR_ROOT | grep "^v3m4_c" | sort)
  python3 $IN/pair_table.py $labels > $D/table_v3m4.txt 2>&1
  python3 $READ/paired_by_prompt.py $labels > $D/paired_v3m4.txt 2>&1
  python3 $READ/first_audio_tail.py $labels > $D/first_audio_tail_v3m4.txt 2>&1
else
  echo "v3m4 pairs skipped: units failed" >> $L
fi
echo "queue_v3_landing_done $(date +%T)" >> $L
