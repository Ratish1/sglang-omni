#!/usr/bin/env bash
# One PR branch against main f31cf9e9a on cards 2 and 3: its touched unit suites, a c1 pair (seeded when the PR moves
# data only and must be byte identical, unseeded when it changes bits), then c16 and c64 pairs in both card orders with
# each arm's SM Issue, full corpus, WER and SIM. Waits for a marker in the queue log, so runs chain one after another.
# usage: queue_pr_pairs.sh <name> <PR tree> <marker to wait for> <seeded|unseeded> "<unit paths>"
set -u
export PAIR_ROOT=/data/q3t_20261021/pairs
D=/data/q3t_20261021
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
M=/workspace/main_f31cf9e9a
NAME=$1
PR=$2
WAIT=$3
SEEDED=$4
UNITS=$5
L=$D/queue.log
mkdir -p $D/units $PAIR_ROOT
until grep -q "$WAIT" $L; do sleep 30; done
echo "queue $NAME start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) pr $(git -C $PR rev-parse HEAD:sglang_omni)" >> $L
# the suites' GPU tests use cards 0 and 1 by name, and those are another container's; none of these suites reads
# the variable (only tests/unit_test/pipeline's wiring tests do)
(cd $PR && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$PR timeout 3600 python3 -m pytest $UNITS -q -p no:cacheprovider -rfEs \
  > $D/units/$NAME.log 2>&1)
echo "units $NAME rc $? $(grep -E 'passed|failed' $D/units/$NAME.log | tail -n 1) $(date +%T)" >> $L
run_pair() {
  python3 $IN/pair.py --label $1 --tree $M --b-tree $PR --concurrency $2 --a-card $3 --b-card $4 --port $5 ${6:-} \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) $(grep -h 'byte identical' $D/$1.out) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)" >> $L
}
if [ "$SEEDED" = seeded ]; then
  run_pair ${NAME}_c1_seed 1 2 3 7000 "--seed 1234"
else
  run_pair ${NAME}_c1 1 2 3 7000
fi
run_pair ${NAME}_c16_a 16 2 3 13000 --gpu-metrics
run_pair ${NAME}_c16_b 16 3 2 18500 --gpu-metrics
run_pair ${NAME}_c64_a 64 2 3 22000 --gpu-metrics
run_pair ${NAME}_c64_b 64 3 2 30000 --gpu-metrics
python3 $IN/pair_table.py $(ls $PAIR_ROOT | grep "^${NAME}_c" | sort) > $D/table_$NAME.txt 2>&1
echo "queue_${NAME}_done $(date +%T)" >> $L
