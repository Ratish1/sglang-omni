#!/usr/bin/env bash
# After queue_d8v3.sh on the 10-25 lease: D8 version 3's c16 orders read RTF +0.1 and +0.3 % with byte-identical
# seeded output and main's speaker embed cost, so main against main at c16 in both card orders gives this lease's own
# band for them. Unscored.
set -u
D=/data/next1025
export PAIR_ROOT=$D/pairs
K=${T:?the kit tree}/tasks/qwen3-tts-perf/box
M=${M:?the main tree}
L=$D/queue.log
until grep -q queue_d8v3_done $L; do sleep 30; done
echo "queue aa c16 1025 start $(date +%T)" >> $L
for spec in "aa_c16_a 2 3 1200" "aa_c16_b 3 2 3600"; do
  set -- $spec
  python3 $K/v3_20261020/pair.py --label $1 --tree $M --b-tree $M --concurrency 16 --a-card $2 --b-card $3 --port $4 \
    --skip-score > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T) $(python3 $K/v3_20261020/pair_table.py $1 2>&1 | tail -2 | cut -d'|' -f2,7 | tr '\n' ' ')" >> $L
done
python3 $K/v3_20261020/pair_table.py aa_c16_a aa_c16_b > $D/table_aa_c16.txt 2>&1
python3 $K/q3t_20261022/paired_by_prompt.py aa_c16_a aa_c16_b > $D/paired_aa_c16.txt 2>&1
echo "queue_aa_c16_1025_done $(date +%T)" >> $L
