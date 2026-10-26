#!/usr/bin/env bash
# The rest of queue_layout_score.sh, which a watcher stopped early: the six 1.7B whole-text arms are transcribed on the
# ASR server that is still up, the server stops, similarity runs on every served arm, and the summary is rebuilt from
# the official arms on disk. No official arm reruns. Card 1 only.
set -u
D=${D:?the layout results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
ASR_PGID=${ASR_PGID:?the ASR server process group}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
L=$D/queue.log
echo "queue layout score2 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
for dir in $D/served/17_whole_English $D/served/17_whole_auto $D/served/17_whole_nomask_English \
  $D/served/17_whole_nomask_auto $D/served/17_whole_min2_English $D/served/17_whole_min2_auto; do
  (cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $dir --asr-port 27800 >> $L 2>&1)
  echo "transcribed $(basename $dir) rc $? $(date +%T)" >> $L
done
kill -TERM -$ASR_PGID
for second in $(seq 120); do
  kill -0 $ASR_PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$ASR_PGID 2> /dev/null
echo "asr stopped $(date +%T)" >> $L
DIRS=""
for size in 06 17; do
  for arm in shipped min2_nomask nomask min2 whole whole_nomask whole_min2; do
    for language in English auto; do
      DIRS="$DIRS $D/served/${size}_${arm}_$language"
    done
  done
done
for dir in $DIRS; do
  (cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $dir --sim-card $CARD >> $L 2>&1)
  echo "similarity $(basename $dir) rc $? $(date +%T)" >> $L
done
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py summary $D/official_*.json --served $DIRS --max-new-tokens $CAP \
  > $D/summary.txt 2>&1)
echo "summary2 rc $? $(date +%T)" >> $L
echo "queue_layout_score2_done $(date +%T)" >> $L
