#!/usr/bin/env bash
# After an A/B queue: similarity_natural_stops.py over every arm of D whose scorer did not finish, then ab_table.py
# over D's arms into D/table.txt. Waits for AFTER_LOG to hold AFTER_LINE. Card 0.
set -u
D=${D:?the A/B results directory}
T=${T:?the kit tree}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
L=$D/rescore.log
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
echo "queue rescore start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
labels=$(ls $D/arms)
python3 $HERE/similarity_natural_stops.py $D/arms $labels --card 0 >> $L 2>&1
echo "rescored rc $? $(date +%T)" >> $L
python3 $HERE/ab_table.py $D/arms $labels > $D/table.txt 2>&1
echo "queue_rescore_done $(date +%T)" >> $L
