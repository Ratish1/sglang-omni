#!/usr/bin/env bash
# The 10-22 chain against upstream main 1b726ab64 on cards 2 and 3 (0 and 1 are the CosyVoice container's): the touched
# unit suites and the real-checkpoint gates on the chain head, then SeedTTS EN pairs main against the chain at c16 and
# c64 in both card orders with SM Issue and an unseeded c1 pair, the vocoder-only quality pair with the codes held equal
# (seeded c1 both orders, main against the D2 head, which carries D4 and D2 only), D10 alone at c64 in both orders,
# and the long cells L1 c16, L3 c16 and L2 c64, then the seeded L1 c16 pair for D1 (its parent against it).
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
SEEDED=$T/tasks/qwen3-tts-perf/box/v3_landing_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
M=/workspace/main_1b726ab64
C=${C:?the chain head tree}
D2=/workspace/c22_d9d73ccd5
PRE_D10=/workspace/c22_8a6e0ffc8
D10=/workspace/c22_94a7ba7dc
PRE_D1=/workspace/c22_d124c75c8
D1=/workspace/c22_1cb7cc3c8
O=tests/unit_test/qwen3_omni
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
mkdir -p $D/units $PAIR_ROOT
echo "queue final start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')chain $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
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
    tests/unit_test/qwen3_tts/test_pipeline.py -m benchmark -k "test_real_" -q -p no:cacheprovider -rfEs \
    > $D/units/gates_$1.log 2>&1)
  echo "gates $1 rc $? $(grep -E 'passed|failed' $D/units/gates_$1.log | tail -n 1) $(date +%T)" >> $L
}
run_pair() {
  local label=$1 a=$2 b=$3
  shift 3
  python3 $IN/pair.py --label $label --tree $a --b-tree $b "$@" > $D/$label.out 2>&1
  echo "$label rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log) $(grep 'byte identical' $D/$label.out)" >> $L
}
units chain $C "tests/unit_test/qwen3_tts tests/unit_test/kernels $O/test_code2wav.py $O/test_code2wav_cuda_graph.py $O/test_code2wav_batching.py $O/test_code2wav_overlap.py $O/test_code2wav_snake_beta.py"
CHAIN_UNITS=$?
gates chain $C
if [ $CHAIN_UNITS -ne 0 ]; then
  echo "pairs skipped: units failed" >> $L
  echo "queue_final_done $(date +%T)" >> $L
  exit 1
fi
run_pair fin_c16_a $M $C --concurrency 16 --a-card 2 --b-card 3 --port 7000 --gpu-metrics
run_pair fin_c16_b $M $C --concurrency 16 --a-card 3 --b-card 2 --port 7600 --gpu-metrics
run_pair fin_c64_a $M $C --concurrency 64 --a-card 2 --b-card 3 --port 8200 --gpu-metrics
run_pair fin_c64_b $M $C --concurrency 64 --a-card 3 --b-card 2 --port 8800 --gpu-metrics
run_pair fin_c1 $M $C --concurrency 1 --a-card 2 --b-card 3 --port 10000
labels=$(ls $PAIR_ROOT | grep "^fin_c" | sort)
python3 $IN/pair_table.py $labels > $D/table_fin.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_fin.txt 2>&1
python3 $READ/first_audio_tail.py $labels > $D/first_audio_tail_fin.txt 2>&1
run_pair fins_c1_a $M $D2 --concurrency 1 --a-card 2 --b-card 3 --port 10600 --seed 1234
run_pair fins_c1_b $M $D2 --concurrency 1 --a-card 3 --b-card 2 --port 11200 --seed 1234
python3 $SEEDED/seeded_quality.py fins_c1_a fins_c1_b > $D/seeded_quality_d2.txt 2>&1
run_pair fin_d10_c64_a $PRE_D10 $D10 --concurrency 64 --a-card 2 --b-card 3 --port 11800
run_pair fin_d10_c64_b $PRE_D10 $D10 --concurrency 64 --a-card 3 --b-card 2 --port 12400
python3 $IN/pair_table.py fin_d10_c64_a fin_d10_c64_b > $D/table_d10.txt 2>&1
run_pair fin_L1_c16 $M $C --concurrency 16 --a-card 2 --b-card 3 --port 13000 --meta $LONG/L1_c16_run.lst
run_pair fin_L3_c16 $M $C --concurrency 16 --a-card 3 --b-card 2 --port 13600 --meta $LONG/L3_c16_run.lst
run_pair fin_L2_c64 $M $C --concurrency 64 --a-card 2 --b-card 3 --port 14200 --meta $LONG/L2_c64_run.lst
python3 $IN/pair_table.py fin_L1_c16 fin_L3_c16 fin_L2_c64 > $D/table_long.txt 2>&1
run_pair fins_L1_c16 $PRE_D1 $D1 --concurrency 16 --a-card 2 --b-card 3 --port 14800 --meta $LONG/L1_c16_run.lst --seed 1234
python3 $SEEDED/seeded_quality.py fins_L1_c16 > $D/seeded_quality_d1.txt 2>&1
echo "queue_final_done $(date +%T)" >> $L
