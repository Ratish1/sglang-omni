#!/usr/bin/env bash
# The B profile: main 204ba29a5 (t_m, card 0) and B (t_b, card 1) traced at the same time by
# cosy_profile_run_c20261013.sh (probe ranges, graph level CUDA graph trace, OS runtime, c1 seeded and c16 cells,
# py-spy at c16), after A's queue has released both cards.
# usage (in the container): bash queue_prof_c20261013.sh
R=/data/c20261013
IN=$R/in
OUT=$R/prof
mkdir -p $OUT
while pgrep -f queue_a_c20261013.sh > /dev/null; do sleep 30; done
echo "start $(date +%T)" >> $OUT/queue.txt
bash $IN/cosy_profile_run_c20261013.sh t_m 0 0-15,64-79 43000 $OUT/t_m &
bash $IN/cosy_profile_run_c20261013.sh t_b 1 16-31,80-95 43200 $OUT/t_b &
wait
echo "queue done $(date +%T)" >> $OUT/queue.txt
