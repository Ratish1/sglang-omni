#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: L3's static c16 gap cost. With every reference cached (p_c1) neither arm
# tokenizes while measured, and L3 still reads inter chunk p95 +4.1 ; +4.3 %, req/s -0.4 ; -1.1 %. t_l2c and t_l3c
# (the measurement only cache patch on L2 and on main + L2 + L3) profiled together under nsys with the probe, PRIME=1
# (an untraced pass over the full set fills the cache), then one traced c16 cell of 320 requests, all cached; the
# ledgers (AR by rows, request states), the chunk timeline and the device overlap split over both traces.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_p.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_p_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_p.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
log "queue p start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
rm -f $D/prof_l2c/ready $D/prof_l3c/ready
PRIME=1 PYSPY=0 CELLS="c16:16:320" PEER=$D/prof_l3c bash $IN/cosy_profile_run_c20261021.sh t_l2c 0 $(cpus 0) 47000 $D/prof_l2c &
PRIME=1 PYSPY=0 CELLS="c16:16:320" PEER=$D/prof_l2c bash $IN/cosy_profile_run_c20261021.sh t_l3c 1 $(cpus 1) 47400 $D/prof_l3c &
wait
for arm in l2c l3c; do
  log "prof_$arm $(tail -n 1 $D/prof_$arm/progress.txt) prime $(grep 'bench prime' $D/prof_$arm/progress.txt)"
done
(cd $IN && python3 chunk_timeline_c20261018.py l2c=$D/prof_l2c/c16.sqlite l3c=$D/prof_l3c/c16.sqlite --skip-head-s 1 --skip-tail-s 1) \
  > $D/chunk_timeline_l2c_l3c.txt 2>&1
python3 $IN/device_overlap_c20261022.py l2c=$D/prof_l2c/c16.sqlite l3c=$D/prof_l3c/c16.sqlite > $D/device_overlap_l2c_l3c.txt 2>&1
log "analyses rc $?"
log "queue p done"
