#!/usr/bin/env bash
# 10-22, the 0.5.22 container, cards 0 and 1, after queue v: L3's c16 gaps (p_r3: mean +2.2 ; +2.8 %, p95 +5.6 ; +4.3 %
# in both orders). t_l2 on card 0 and t_l3 on card 1 profiled together under nsys with the probe (graph level CUDA
# graph tracing, the OS runtime), one traced c16 cell of 320 requests, then py-spy at c16; the ledgers, then the
# tokenizer contention script over both traces (AR periods and vocoder steps split by overlap with pre.tokenize).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022r/queue_w.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_w_c20261022.sh'
D=/data/c20261022r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }
until grep -q "queue v done" $D/queue_r.txt; do sleep 30; done
log "queue w start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
rm -f $D/prof_l2/ready $D/prof_l3/ready
CELLS="c16:16:320" PEER=$D/prof_l3 bash $IN/cosy_profile_run_c20261021.sh t_l2 0 0-15,64-79 47000 $D/prof_l2 &
CELLS="c16:16:320" PEER=$D/prof_l2 bash $IN/cosy_profile_run_c20261021.sh t_l3 1 16-31,80-95 47400 $D/prof_l3 &
wait
for arm in l2 l3; do
  log "prof_$arm $(tail -n 1 $D/prof_$arm/progress.txt) labels $(grep -c . $D/prof_$arm/labels_c16.txt 2> /dev/null)"
done
python3 $IN/tokenizer_contention_c20261020.py l2=$D/prof_l2/c16.sqlite l3=$D/prof_l3/c16.sqlite > $D/contention_l2_l3.txt 2>&1
log "contention rc $? $(head -c 600 $D/contention_l2_l3.txt | tr '\n' ' ')"
log "queue w done"
