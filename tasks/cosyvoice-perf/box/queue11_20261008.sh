#!/usr/bin/env bash
# After queue10: a third pair of the tokenizer fix (items 1 and 2 against items 1 and 2 plus the fix),
# stream and buffered c16, to read its inter chunk mean over three pairs.
cd /data/p20261008
P=/data/p20261008
until grep -q "queue10 done" $P/queue.txt 2>/dev/null; do sleep 10; done
echo "abtok3 $(date +%T)" >> $P/queue.txt
CELLS="c16:16:::1 b16:16:::0" SCORE_CELLS=none bash ab_pair.sh $P/abtok3 convk tokarena 23000
echo "queue11 done $(date +%T)" >> $P/queue.txt
