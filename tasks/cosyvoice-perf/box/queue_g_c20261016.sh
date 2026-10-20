#!/usr/bin/env bash
# 10-16 second box, after queue_f: the profile of the stack head (HEAD below) against A3 at the same time, nsys with
# the probe and graph level tracing, c16 of 320 and c64 of 640, py-spy at c16, the
# ledgers on the box (box/cosy_profile_run_c20261016.sh, on the box as in/cosy_profile_run_v2_c20261016.sh). The
re-rank before the next item.
# usage (in the container): HEAD=<tree> setsid bash -c 'echo $$ > queue_g.pgid; exec bash queue_g_c20261016.sh'
D=/data/c20261016
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_g.txt; }

log "queue start head $HEAD"
export CELLS="c16:16:320 c64:64:640"
PEER=$D/g_head bash $IN/cosy_profile_run_v2_c20261016.sh t_a3 2 0-15,64-79 22000 $D/g_a3 &
PEER=$D/g_a3 bash $IN/cosy_profile_run_v2_c20261016.sh $HEAD 3 16-31,80-95 23000 $D/g_head
wait
log "queue done"
