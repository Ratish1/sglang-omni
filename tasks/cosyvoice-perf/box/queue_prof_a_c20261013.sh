#!/usr/bin/env bash
# A's profile: main 204ba29a5 (t_m, card 0) and A (t_a, card 1) traced at the same time by
# cosy_profile_run_c20261013.sh (probe ranges, graph level CUDA graph trace, OS runtime, c1 seeded and c16 cells,
# py-spy at c16), each arm waiting for the other's health check before its benches.
# usage (in the container): bash queue_prof_a_c20261013.sh
R=/data/c20261013
IN=$R/in
OUT=$R/prof_a
mkdir -p $OUT
echo "start $(date +%T)" >> $OUT/queue.txt
PEER=$OUT/t_a bash $IN/cosy_profile_run_c20261013.sh t_m 0 0-15,64-79 43400 $OUT/t_m &
PEER=$OUT/t_m bash $IN/cosy_profile_run_c20261013.sh t_a 1 16-31,80-95 43600 $OUT/t_a &
wait
echo "queue done $(date +%T)" >> $OUT/queue.txt
