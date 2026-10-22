#!/usr/bin/env bash
# D9 on the chain (the vocoder's last conv on the single output conv kernel): the touched unit suites and the gates on
# the D9 head, the vocoder bench of the chain before D9 against D9 (card 3, both orders), the vocoder-only quality with
# the codes held equal (seeded c1, seeds 1234 and 4321, before D9 against D9), then SeedTTS EN pairs main against the
# D9 head at c16 and c64 in both card orders with SM Issue and an unseeded c1 pair.
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=/workspace/main_1b726ab64
PRE_D9=${PRE_D9:?the chain before D9}
C=${C:?the D9 head tree}
O=tests/unit_test/qwen3_omni
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
echo "queue d9 start $(date +%T) pre $(git -C $PRE_D9 rev-parse HEAD:sglang_omni) d9 $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $C && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$C timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/kernels \
  $O/test_code2wav.py $O/test_code2wav_cuda_graph.py $O/test_code2wav_batching.py $O/test_code2wav_overlap.py \
  $O/test_code2wav_snake_beta.py -q -p no:cacheprovider -rfEs > $D/units/d9.log 2>&1)
UNITS=$?
echo "units d9 rc $UNITS $(grep -E 'passed|failed' $D/units/d9.log | tail -n 1) $(date +%T)" >> $L
(cd $C && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$C timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py \
  tests/unit_test/qwen3_tts/test_pipeline.py -m benchmark -k "test_real_" -q -p no:cacheprovider -rfEs \
  > $D/units/gates_d9.log 2>&1)
echo "gates d9 rc $? $(grep -E 'passed|failed' $D/units/gates_d9.log | tail -n 1) $(date +%T)" >> $L
if [ $UNITS -ne 0 ]; then
  echo "stopped: units failed" >> $L
  echo "queue_d9_done $(date +%T)" >> $L
  exit 1
fi
for arm in "pre $PRE_D9" "d9 $C" "d9 $C" "pre $PRE_D9"; do
  set -- $arm
  echo "== bench $1" >> $D/bench_d9.log
  PYTHONPATH=$2 python3 $HERE/vocoder_decode_bench.py --card 3 2>&1 | grep -E "^(cold|warm|tail) " >> $D/bench_d9.log
done
echo "bench d9 done $(date +%T)" >> $L
run_pair() {
  local label=$1 a=$2 b=$3
  shift 3
  python3 $IN/pair.py --label $label --tree $a --b-tree $b "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(grep 'byte identical' $D/$label.out)" >> $L
  if [ "$failed_a" != 0 ] || [ "$failed_b" != 0 ]; then
    echo "stopped: capture or compile failures in $label" >> $L
    echo "queue_d9_done $(date +%T)" >> $L
    exit 1
  fi
}
run_pair d9s_c1_1234 $PRE_D9 $C --concurrency 1 --a-card 2 --b-card 3 --port 17000 --seed 1234
run_pair d9s_c1_4321 $PRE_D9 $C --concurrency 1 --a-card 3 --b-card 2 --port 18200 --seed 4321
python3 $SEEDED/seeded_quality.py d9s_c1_1234 d9s_c1_4321 > $D/seeded_quality_d9.txt 2>&1
run_pair d9_c16_a $M $C --concurrency 16 --a-card 2 --b-card 3 --port 19400 --gpu-metrics
run_pair d9_c16_b $M $C --concurrency 16 --a-card 3 --b-card 2 --port 20600 --gpu-metrics
run_pair d9_c64_a $M $C --concurrency 64 --a-card 2 --b-card 3 --port 21800 --gpu-metrics
run_pair d9_c64_b $M $C --concurrency 64 --a-card 3 --b-card 2 --port 23000 --gpu-metrics
run_pair d9_c1 $M $C --concurrency 1 --a-card 2 --b-card 3 --port 24200
labels=$(ls $PAIR_ROOT | grep "^d9_c" | sort)
python3 $IN/pair_table.py $labels > $D/table_d9.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_d9.txt 2>&1
python3 $READ/first_audio_tail.py $labels > $D/first_audio_tail_d9.txt 2>&1
echo "queue_d9_done $(date +%T)" >> $L
