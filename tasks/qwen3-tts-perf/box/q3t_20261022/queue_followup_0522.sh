#!/usr/bin/env bash
# After the landing queue: score the arms whose scorer failed (two ASR servers started together raced for one
# rendezvous port), one at a time, then two more P6 c64 pairs, one in each card order, since p6m3's two c64 orders
# disagreed (-1.6 and +0.2 % RTF).
set -u
export PAIR_ROOT=/data/q3t_20261022/pairs
D=/data/q3t_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
M=/workspace/main_057d97629
P6=/workspace/p6pr_afceccbb0
L=$D/queue.log
until grep -q queue_landing_0522_done $L; do sleep 30; done
echo "queue followup 0522 start $(date +%T)" >> $L
for arm_dir in $PAIR_ROOT/*/a $PAIR_ROOT/*/b; do
  if [ ! -f $arm_dir/score--transcribe-only.log ]; then
    card=$(head -n 1 $arm_dir/tree.txt | cut -d' ' -f3)
    (cd $M && PYTHONPATH=$IN:$M python3 -c "import sys; from pathlib import Path; import pair; pair.score(Path(sys.argv[1]), int(sys.argv[2]), 9600, sys.argv[3])" \
      $arm_dir $card $M > $arm_dir/rescore.log 2>&1)
    echo "rescore $arm_dir card $card rc $? $(date +%T)" >> $L
  else
    :
  fi
done
run_pair() {
  python3 $IN/pair.py --label $1 --tree $M --b-tree $P6 --concurrency 64 --a-card $2 --b-card $3 --port $4 --gpu-metrics \
    > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) capture or compile failures a $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/a/serve.log) b $(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$1/b/serve.log)" >> $L
}
run_pair p6m3_c64_c 2 3 9000
run_pair p6m3_c64_d 3 2 15000
python3 $IN/pair_table.py $(ls $PAIR_ROOT | grep "^p6m3_c" | sort) > $D/table_p6m3.txt 2>&1
echo "queue_followup_0522_done $(date +%T)" >> $L
