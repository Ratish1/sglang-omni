#!/usr/bin/env bash
# A stacked branch against the branch it sits on, on cards 2 and 3: its touched unit suites and the real-checkpoint
# gates (stopping there on a failure), then an unseeded c1 pair and c16 and c64 pairs in both card orders with SM Issue.
# usage: queue_stack_pairs.sh <name> <base tree> <branch tree> <marker to wait for> "<unit paths>"
set -u
export PAIR_ROOT=/data/q3t_20261021/pairs
D=/data/q3t_20261021
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
SNAP=/data/hf/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3
NAME=$1
BASE=$2
PR=$3
WAIT=$4
UNITS=$5
L=$D/queue.log
mkdir -p $D/units $PAIR_ROOT
until grep -q "$WAIT" $L; do sleep 30; done
echo "queue $NAME start $(date +%T) base $(git -C $BASE rev-parse HEAD:sglang_omni) pr $(git -C $PR rev-parse HEAD:sglang_omni)" >> $L
# the suites' GPU tests use cards 0 and 1 by name, and those are another container's
(cd $PR && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$PR timeout 3600 python3 -m pytest $UNITS -q -p no:cacheprovider -rfEs \
  > $D/units/$NAME.log 2>&1)
echo "units $NAME rc $? $(grep -E 'passed|failed' $D/units/$NAME.log | tail -n 1) $(date +%T)" >> $L
(cd $PR && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
  PYTHONPATH=$PR timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py \
  tests/unit_test/qwen3_tts/test_pipeline.py -k "test_real_" -q -p no:cacheprovider -rfEs > $D/units/${NAME}_gates.log 2>&1)
echo "gates $NAME rc $? $(grep -E 'passed|failed' $D/units/${NAME}_gates.log | tail -n 1) $(date +%T)" >> $L
if grep -qE "[0-9]+ failed|[0-9]+ error" $D/units/$NAME.log $D/units/${NAME}_gates.log; then
  echo "queue_${NAME}_stopped units or gates failed $(date +%T)" >> $L
  exit 1
else
  :
fi
run_pair() {
  python3 $IN/pair.py --label $1 --tree $BASE --b-tree $PR --concurrency $2 --a-card $3 --b-card $4 --port $5 ${6:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)" >> $L
}
run_pair ${NAME}_c1 1 2 3 7000
run_pair ${NAME}_c16_a 16 2 3 13000 --gpu-metrics
run_pair ${NAME}_c16_b 16 3 2 18500 --gpu-metrics
run_pair ${NAME}_c64_a 64 2 3 22000 --gpu-metrics
run_pair ${NAME}_c64_b 64 3 2 30000 --gpu-metrics
python3 $IN/pair_table.py $(ls $PAIR_ROOT | grep "^${NAME}_c" | sort) > $D/table_$NAME.txt 2>&1
echo "queue_${NAME}_done $(date +%T)" >> $L
