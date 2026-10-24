#!/usr/bin/env bash
# 10-24, cards 0 and 1, after queue_aa2.sh: L6's c1 inter chunk. Each seeded c1f request against itself reads L6's
# inter chunk mean +0.86 ; +0.30 ms over L4's (L4 against main reads +0.20 ; -1.05 ms, so about +0.6 ms is L6's), while
# its first audio is -1.0 ; -1.5 ms. L6 gives a frame tier min(max_rows, frames // min_hop_frames) row slots, so a one
# row later hop of 200 frames replays four slots where L4 replays one. The nsys probe pass of t_l4p and t_l6p at once,
# seeded c1 over 160 requests, then each arm's replays by tier with their own device time (graph_padding_c20261020.py)
# and the cosy ledger's step lines.
# Decision rule, written before the run: the cost is L6's replay layout when its one row hops' prefix graph device
# time per replay exceeds L4's at the same hop sizes.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l65/queue_l6c1.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l6c1.sh'
D=/data/c20261024
P=$D/l65
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q 'queue aa2 done' $P/queue.txt; do sleep 30; done
CELLS="c1:1:160:1234" PYSPY=0 PEER=$P/nsys_c1_t_l6p \
  bash $IN/cosy_profile_run_c20261021.sh t_l4p 0 $(cpus 0) 47000 $P/nsys_c1_t_l4p &
CELLS="c1:1:160:1234" PYSPY=0 PEER=$P/nsys_c1_t_l4p \
  bash $IN/cosy_profile_run_c20261021.sh t_l6p 1 $(cpus 1) 47400 $P/nsys_c1_t_l6p &
wait
log "nsys c1 t_l4p $(tail -n 1 $P/nsys_c1_t_l4p/progress.txt) t_l6p $(tail -n 1 $P/nsys_c1_t_l6p/progress.txt)"
for name in t_l4p t_l6p; do
  python3 $IN/graph_padding_c20261020.py $P/nsys_c1_$name/c1.sqlite $P/nsys_c1_$name/serve.log --skip-head-s 1 \
    --skip-tail-s 1 > $P/graph_padding_c1_$name.txt 2>&1
  log "graph padding c1 $name rc $?"
done
log "queue l6c1 done"
