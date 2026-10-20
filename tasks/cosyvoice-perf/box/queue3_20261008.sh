#!/usr/bin/env bash
# After queue2_20261008.sh: py-spy GIL ownership of the full chain at stream c16 on card 0, and the
# HiFT final trace pair over long histories on main on card 1.
# usage (in the container): bash queue3_20261008.sh
cd /data/p20261008
P=/data/p20261008
until grep -q "queue2 done" $P/queue.txt 2>/dev/null; do sleep 10; done
echo "queue3 $(date +%T)" >> $P/queue.txt
bash pyspy_run.sh mixed 0 28000 $P/pyspy_mixed &
SPY=$!
bash trace_pairs.sh sglang-omni 1 $P/tp_main_hift_long hiftstep_final16_long hiftstep_final1_long &
TRACE=$!
wait $SPY $TRACE
echo "queue3 done $(date +%T)" >> $P/queue.txt
