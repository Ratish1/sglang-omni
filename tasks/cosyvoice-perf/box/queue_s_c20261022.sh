#!/usr/bin/env bash
# 10-22, the 0.5.22 container, cards 0 and 1, after queue r's phase b: the card memory of t_l1 (#3019) against t_l2
# (L2) through a same time c64 pair, both orders (p_r5): nvidia-smi dmon samples every card's framebuffer each
# second, and the time each order's c64 bench starts is kept, so each server's memory at the start of serving
# reads against its peak while serving.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022r/queue_s.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_s_c20261022.sh'
D=/data/c20261022r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }
until grep -q "queue b done" $D/queue_r.txt; do sleep 30; done
log "queue s start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
nvidia-smi dmon -i 0,1 -s m -d 1 -o T > $D/p_r5_dmon.log 2>&1 &
dmon=$!
for order in a b; do
  (until [ -e $D/p_r5/$order/bench_gpu0_c64.log ]; do sleep 1; done; echo "$order $(date +%T)" >> $D/p_r5_bench_start.txt) &
done
(cd /workspace/sglang-omni && PAIR_CELLS=c64 python3 $IN/serve_pair_c20261021.py /workspace/t_l1 /workspace/t_l2 $D/p_r5 \
  > $D/p_r5.log 2>&1)
log "pair p_r5 rc $?"
kill $dmon
log "queue s done bench starts $(tr '\n' ' ' < $D/p_r5_bench_start.txt)"
