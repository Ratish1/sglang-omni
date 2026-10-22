#!/usr/bin/env bash
# After queue_d8b.sh: D8 part 2 whole (the reference encode's recurring widths and the speaker embedding past its graphs
# on the device at recurring widths). The speaker bench on L3 c16's references before and after (card 3) and their
# embeddings compared; the qwen3_tts units on the head; the long cells D9 against D8 part 2 whole (L1 c1, L1 c16 and
# L3 c16 in both card orders); then the long cells main against the head (L1 c16, L3 c16, L2 c64).
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
M=/workspace/main_1b726ab64
PRE=${PRE:?the chain at D9}
MID=${MID:?the chain with the reference encode part}
C=${C:?the D8 part 2 head tree}
L=$D/queue.log
until grep -q queue_d8b_done $L; do sleep 20; done
echo "queue d8c start $(date +%T) pre $(git -C $PRE rev-parse HEAD:sglang_omni) mid $(git -C $MID rev-parse HEAD:sglang_omni) head $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for arm in "mid $MID" "head $C"; do
  set -- $arm
  echo "== speaker bench $1" >> $D/speaker_bench_d8c.log
  PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_d8c_$1.pt --card 3 --repeats 5 --meta $LONG/L3_c16_run.lst 2>&1 \
    | grep -E "samples, first|embed wall|graphs:|Error|Traceback" >> $D/speaker_bench_d8c.log
done
PYTHONPATH=$C python3 $HERE/speaker_embed_bench.py --compare $D/speaker_d8c_mid.pt $D/speaker_d8c_head.pt >> $D/speaker_bench_d8c.log 2>&1
echo "speaker bench d8c done $(date +%T)" >> $L
(cd $C && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$C timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfEs \
  > $D/units/d8c.log 2>&1)
UNITS=$?
echo "units d8c rc $UNITS $(grep -E 'passed|failed' $D/units/d8c.log | tail -n 1) $(date +%T)" >> $L
if [ $UNITS -ne 0 ]; then
  echo "stopped: units failed" >> $L
  echo "queue_d8c_done $(date +%T)" >> $L
  exit 1
fi
run_pair() {
  local label=$1 a=$2 b=$3
  shift 3
  python3 $IN/pair.py --label $label --tree $a --b-tree $b "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
}
run_pair d8c_L1_c1 $PRE $C --concurrency 1 --a-card 2 --b-card 3 --port 25400 --meta $LONG/L1_c1_run.lst
run_pair d8c_L1_c16_a $PRE $C --concurrency 16 --a-card 2 --b-card 3 --port 26600 --meta $LONG/L1_c16_run.lst
run_pair d8c_L1_c16_b $PRE $C --concurrency 16 --a-card 3 --b-card 2 --port 27800 --meta $LONG/L1_c16_run.lst
run_pair d8c_L3_c16_a $PRE $C --concurrency 16 --a-card 2 --b-card 3 --port 12000 --meta $LONG/L3_c16_run.lst
run_pair d8c_L3_c16_b $PRE $C --concurrency 16 --a-card 3 --b-card 2 --port 13200 --meta $LONG/L3_c16_run.lst
labels=$(ls $PAIR_ROOT | grep "^d8c_" | sort)
python3 $IN/pair_table.py $labels > $D/table_d8c.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_d8c.txt 2>&1
run_pair finl_L1_c16 $M $C --concurrency 16 --a-card 2 --b-card 3 --port 10000 --meta $LONG/L1_c16_run.lst
run_pair finl_L3_c16 $M $C --concurrency 16 --a-card 3 --b-card 2 --port 22000 --meta $LONG/L3_c16_run.lst
run_pair finl_L2_c64 $M $C --concurrency 64 --a-card 2 --b-card 3 --port 23200 --meta $LONG/L2_c64_run.lst
labels=$(ls $PAIR_ROOT | grep "^finl_" | sort)
python3 $IN/pair_table.py $labels > $D/table_long_final.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_long_final.txt 2>&1
echo "queue_d8c_done $(date +%T)" >> $L
