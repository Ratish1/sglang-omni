#!/usr/bin/env bash
# 10-22, host-85-234-79-221: queue p with the cards swapped, t_l3c on card 0 and t_l2c on card 1, so the vocoder's
# graph device times of the two arms compare across both cards (queue p: L3c on card 1 read its HiFT and whole history
# graphs 1.4 to 6.3 % longer per step, while the A/A p_y1 read card 1 0.9 to 1.8 % slower in gaps). Cached references,
# one traced c16 cell of 320 requests, the ledgers, the chunk timeline.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_q.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_q_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_q.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
log "queue q start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
rm -f $D/prof_l2c_b/ready $D/prof_l3c_b/ready
PRIME=1 PYSPY=0 CELLS="c16:16:320" PEER=$D/prof_l2c_b bash $IN/cosy_profile_run_c20261021.sh t_l3c 0 $(cpus 0) 47000 $D/prof_l3c_b &
PRIME=1 PYSPY=0 CELLS="c16:16:320" PEER=$D/prof_l3c_b bash $IN/cosy_profile_run_c20261021.sh t_l2c 1 $(cpus 1) 47400 $D/prof_l2c_b &
wait
for arm in l2c_b l3c_b; do
  log "prof_$arm $(tail -n 1 $D/prof_$arm/progress.txt) prime $(grep 'bench prime' $D/prof_$arm/progress.txt)"
done
(cd $IN && python3 chunk_timeline_c20261018.py l2c=$D/prof_l2c_b/c16.sqlite l3c=$D/prof_l3c_b/c16.sqlite --skip-head-s 1 --skip-tail-s 1) \
  > $D/chunk_timeline_l2c_l3c_b.txt 2>&1
log "queue q done"
