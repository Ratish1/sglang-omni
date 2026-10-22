#!/usr/bin/env bash
# D8 part 2 on the chain (eager reference encodes past the graph ladder rounded up to its largest bucket): the
# qwen3_tts unit suite on its head, then the long-reference cells it touches, the chain before against D8 part 2:
# L1 c1, L1 c16 and L3 c16 in both card orders. SeedTTS EN references are at most 111 frames, all on graphs.
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
PRE=${PRE:?the chain before D8 part 2}
C=${C:?the D8 part 2 head tree}
L=$D/queue.log
echo "queue d8b start $(date +%T) pre $(git -C $PRE rev-parse HEAD:sglang_omni) d8b $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $C && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$C timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider -rfEs \
  > $D/units/d8b.log 2>&1)
UNITS=$?
echo "units d8b rc $UNITS $(grep -E 'passed|failed' $D/units/d8b.log | tail -n 1) $(date +%T)" >> $L
if [ $UNITS -ne 0 ]; then
  echo "stopped: units failed" >> $L
  echo "queue_d8b_done $(date +%T)" >> $L
  exit 1
fi
run_pair() {
  local label=$1
  shift
  python3 $IN/pair.py --label $label --tree $PRE --b-tree $C "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
}
run_pair d8b_L1_c1 --concurrency 1 --a-card 2 --b-card 3 --port 25400 --meta $LONG/L1_c1_run.lst
run_pair d8b_L1_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 26600 --meta $LONG/L1_c16_run.lst
run_pair d8b_L1_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 27800 --meta $LONG/L1_c16_run.lst
run_pair d8b_L3_c16_a --concurrency 16 --a-card 2 --b-card 3 --port 12000 --meta $LONG/L3_c16_run.lst
run_pair d8b_L3_c16_b --concurrency 16 --a-card 3 --b-card 2 --port 13200 --meta $LONG/L3_c16_run.lst
labels=$(ls $PAIR_ROOT | grep "^d8b_" | sort)
python3 $IN/pair_table.py $labels > $D/table_d8b.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_d8b.txt 2>&1
echo "queue_d8b_done $(date +%T)" >> $L
