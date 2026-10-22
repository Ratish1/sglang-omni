#!/usr/bin/env bash
# After queue_d8b.sh: the long-reference cells main against the chain head with D9 and D8 part 2, L1 c16, L3 c16 and
# L2 c64, one order each as the chain's first long cells ran.
set -u
export PAIR_ROOT=/data/chain_20261022/pairs
D=/data/chain_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
READ=$T/tasks/qwen3-tts-perf/box/q3t_20261022
LONG=/data/prof_20261022/formal/meta_w2/long
M=/workspace/main_1b726ab64
C=${C:?the chain head tree}
L=$D/queue.log
until grep -q queue_d8b_done $L; do sleep 20; done
echo "queue long final start $(date +%T) chain $(git -C $C rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
run_pair() {
  local label=$1
  shift
  python3 $IN/pair.py --label $label --tree $M --b-tree $C "$@" > $D/$label.out 2>&1
  local rc=$?
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b" >> $L
}
run_pair finl_L1_c16 --concurrency 16 --a-card 2 --b-card 3 --port 10000 --meta $LONG/L1_c16_run.lst
run_pair finl_L3_c16 --concurrency 16 --a-card 3 --b-card 2 --port 22000 --meta $LONG/L3_c16_run.lst
run_pair finl_L2_c64 --concurrency 64 --a-card 2 --b-card 3 --port 23200 --meta $LONG/L2_c64_run.lst
labels=$(ls $PAIR_ROOT | grep "^finl_" | sort)
python3 $IN/pair_table.py $labels > $D/table_long_final.txt 2>&1
python3 $READ/paired_by_prompt.py $labels > $D/paired_long_final.txt 2>&1
echo "queue_long_final_done $(date +%T)" >> $L
