#!/usr/bin/env bash
# Similarity only where it decides something, to keep the lease for the branch tests: queue_layout_score2.sh scores the
# 0.6B shipped, proposed (minimum, mask off) and mask-off arms, then stops; this scores the same 1.7B arms and rebuilds
# the summary (the whole-text arms keep WER only). Card 1 only.
set -u
D=${D:?the layout results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
SCORE2_PGID=${SCORE2_PGID:?the score2 process group}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
L=$D/queue.log
until grep -q "^similarity 06_nomask_auto" $L; do
  sleep 2
done
kill -TERM -$SCORE2_PGID
echo "queue layout score3 start, score2 stopped after the 0.6B arms $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
for dir in $D/served/17_shipped_English $D/served/17_shipped_auto $D/served/17_min2_nomask_English \
  $D/served/17_min2_nomask_auto $D/served/17_nomask_English $D/served/17_nomask_auto; do
  (cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $dir --sim-card $CARD >> $L 2>&1)
  echo "similarity $(basename $dir) rc $? $(date +%T)" >> $L
done
DIRS=""
for size in 06 17; do
  for arm in shipped min2_nomask nomask min2 whole whole_nomask whole_min2; do
    for language in English auto; do
      DIRS="$DIRS $D/served/${size}_${arm}_$language"
    done
  done
done
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py summary $D/official_*.json --served $DIRS --max-new-tokens $CAP \
  > $D/summary.txt 2>&1)
echo "summary3 rc $? $(date +%T)" >> $L
echo "queue_layout_score2_done $(date +%T)" >> $L
