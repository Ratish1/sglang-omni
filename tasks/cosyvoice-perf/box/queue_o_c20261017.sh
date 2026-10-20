#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: whether the graphs PR needs the ranking once the priority is on main. Tree: item 1
# 425c67687 with the two priority commits and no ranking (t_gp_56b436580). Waits for queue_n; card 3 warms it (its
# smoke test). Pair, both orders: the priority alone on main (t_mp_ed979c580) against it (p_gp, c16 and c64), to be
# read beside p_mpg (the same base against the stack, which has the ranking).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_o.pgid; exec bash queue_o_c20261017.sh'
D=/data/c20261017
IN=$D/in
G=t_gp_56b436580
P=t_mp_ed979c580
log() { echo "$* $(date +%T)" >> $D/queue_o.txt; }

log "queue wait"
until grep -q "queue done" $D/queue_n.txt; do sleep 30; done
log "queue start"
bash $IN/warm_c20261017.sh $G 3 16-31,80-95 22400 $D/smoke_gp
log "smoke gp $(grep 'bench rc' $D/smoke_gp/progress.txt) tracebacks $(grep -c Traceback $D/smoke_gp/serve.log)"
cd /workspace/t_main2
if grep -q "bench rc 0" $D/smoke_gp/progress.txt && ! grep -q Traceback $D/smoke_gp/serve.log; then
  PAIR_CELLS=c16,c64 python3 $IN/serve_pair_c20261017.py /workspace/$P /workspace/$G $D/p_gp > $D/p_gp.log 2>&1
  log "p_gp rc $?"
else
  log "p_gp skipped"
fi
log "queue done"
