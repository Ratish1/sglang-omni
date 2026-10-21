#!/usr/bin/env bash
# The PR branches rebased onto main 057d97629 (SGLang 0.5.22, torch 2.14.1, Triton 3.8.0, flashinfer 0.7.1) on cards 2
# and 3: every branch's touched unit suites and the real-checkpoint gates first, then P6 and V3 against the new main,
# each a c1 pair and c16 and c64 pairs in both card orders with each arm's SM Issue, full corpus, WER and SIM. A branch
# whose units fail gets no pairs.
set -u
export PAIR_ROOT=/data/q3t_20261022/pairs
D=/data/q3t_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
M=/workspace/main_057d97629
P6=/workspace/p6pr_afceccbb0
V3=/workspace/v3pr_15849cd2a
RINGS=/workspace/rings_ae16ad27c
V4=/workspace/v4pr_8c3a6fd09
O=tests/unit_test/qwen3_omni
SNAP=/data/hf/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3
L=$D/queue.log
mkdir -p $D/units $PAIR_ROOT
echo "queue landing 0522 start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) t_marks $(git -C $T rev-parse HEAD)" >> $L
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
  python3 $IN/pair.py --label $1 --tree $M --b-tree $PR --concurrency $2 --a-card $3 --b-card $4 --port $5 ${6:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)" >> $L
}
pairs() {
  NAME=$1
  PR=$2
  echo "queue $NAME start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) pr $(git -C $PR rev-parse HEAD:sglang_omni)" >> $L
  run_pair ${NAME}_c1 1 2 3 7000
  run_pair ${NAME}_c16_a 16 2 3 13000 --gpu-metrics
  run_pair ${NAME}_c16_b 16 3 2 18500 --gpu-metrics
  run_pair ${NAME}_c64_a 64 2 3 22000 --gpu-metrics
  run_pair ${NAME}_c64_b 64 3 2 30000 --gpu-metrics
  python3 $IN/pair_table.py $(ls $PAIR_ROOT | grep "^${NAME}_c" | sort) > $D/table_$NAME.txt 2>&1
  echo "queue_${NAME}_done $(date +%T)" >> $L
}
units p6m3 $P6 "tests/unit_test/qwen3_tts $O/test_talker.py $O/test_predictor_kernels.py"
P6_UNITS=$?
units v3m3 $V3 "tests/unit_test/qwen3_tts $O/test_code2wav.py $O/test_code2wav_cuda_graph.py $O/test_code2wav_batching.py $O/test_code2wav_overlap.py $O/test_code2wav_snake_beta.py"
V3_UNITS=$?
units rings4 $RINGS "tests/unit_test/qwen3_tts"
units v4m3 $V4 "tests/unit_test/qwen3_tts"
for TREE in $M $V3 $RINGS $V4; do
  gates $(basename $TREE) $TREE
done
if [ $P6_UNITS -eq 0 ]; then
  pairs p6m3 $P6
else
  echo "p6m3 pairs skipped: units failed" >> $L
fi
if [ $V3_UNITS -eq 0 ]; then
  pairs v3m3 $V3
else
  echo "v3m3 pairs skipped: units failed" >> $L
fi
echo "queue_landing_0522_done $(date +%T)" >> $L
