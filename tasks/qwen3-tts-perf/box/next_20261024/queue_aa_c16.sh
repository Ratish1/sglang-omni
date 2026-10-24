#!/usr/bin/env bash
# After queue_pairs.sh: D8's two c16 orders read RTF +0.9 and -0.3 %, a wider spread than this lease's near-identical
# code pairs, so main against main at c16 in both card orders gives this node's own band. Unscored.
set -u
D=/data/next1024
export PAIR_ROOT=$D/pairs
K=/workspace/t_marks_2f039db26/tasks/qwen3-tts-perf/box
M=/workspace/main_67476823a
L=$D/queue.log
until grep -q queue_pairs_done $L; do sleep 30; done
echo "queue aa c16 start $(date +%T): main against main at c16 in both card orders, the band for D8's c16 order spread (+0.9 / -0.3 %)" >> $L
for spec in "aa_c16_a 2 3 1200" "aa_c16_b 3 2 3600"; do
  set -- $spec
  python3 $K/v3_20261020/pair.py --label $1 --tree $M --b-tree $M --concurrency 16 --a-card $2 --b-card $3 --port $4 \
    --skip-score > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
done
python3 $K/v3_20261020/pair_table.py aa_c16_a aa_c16_b > $D/table_aa_c16.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py aa_c16_a aa_c16_b > $D/paired_aa_c16.txt 2>&1
echo "queue_aa_c16_done $(date +%T)" >> $L
