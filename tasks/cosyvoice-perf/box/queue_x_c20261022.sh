#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1, container sglang-omni-cosyvoice-0522 on CI's image: L3's c16 gaps
# (p_r3: mean +2.2 ; +2.8 %, p95 +5.6 ; +4.3 % against L2 in both orders). Warms of t_l2 (L2 on main) and t_l3 (main +
# L2 + L3), then both profiled together under nsys with the probe (graph level CUDA graph tracing, the OS runtime),
# one traced c16 cell of 320 requests, then py-spy at c16; the ledgers, then the tokenizer contention script over
# both traces (host side: AR periods and vocoder launch walls split by overlap with pre.tokenize).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_x.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_x_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_x.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
log "queue x start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
for tree in t_l2 t_l3; do
  log "tree $tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-10) sglang_omni $(git -C /workspace/$tree rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
bash $IN/warm_c20261021.sh t_l2 0 $(cpus 0) 50000 $D/warm_t_l2 &
bash $IN/warm_c20261021.sh t_l3 1 $(cpus 1) 50400 $D/warm_t_l3 &
wait
for tree in t_l2 t_l3; do
  log "warm $tree $(grep 'bench rc' $D/warm_$tree/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_$tree/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_$tree/serve.log)"
done
rm -f $D/prof_l2/ready $D/prof_l3/ready
CELLS="c16:16:320" PEER=$D/prof_l3 bash $IN/cosy_profile_run_c20261021.sh t_l2 0 $(cpus 0) 47000 $D/prof_l2 &
CELLS="c16:16:320" PEER=$D/prof_l2 bash $IN/cosy_profile_run_c20261021.sh t_l3 1 $(cpus 1) 47400 $D/prof_l3 &
wait
for arm in l2 l3; do
  log "prof_$arm $(tail -n 1 $D/prof_$arm/progress.txt) labels $(grep -c . $D/prof_$arm/labels_c16.txt 2> /dev/null)"
done
python3 $IN/tokenizer_contention_c20261020.py l2=$D/prof_l2/c16.sqlite l3=$D/prof_l3/c16.sqlite > $D/contention_l2_l3.txt 2>&1
log "contention rc $?"
log "queue x done"
