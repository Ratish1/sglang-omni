#!/usr/bin/env bash
# 10-17 lease, cards 2 and 3: unseeded c1 of the graphs PR's exact tree (t_gp_56b436580) against its base, the
# priority alone on main (t_mp_ed979c580), both orders (p_gp1), so the PR's c1 row is its own tree's and not p_mpg's
# (whose tree has the ranking, which cannot act at c1). Waits for queue_r.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261017/queue_s.pgid; exec bash queue_s_c20261017.sh'
D=/data/c20261017
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_s.txt; }

log "queue wait"
until grep -q "queue done" $D/queue_r.txt; do sleep 30; done
log "queue start"
cd /workspace/t_main2
PAIR_CELLS=c1u python3 $IN/serve_pair_c20261017.py /workspace/t_mp_ed979c580 /workspace/t_gp_56b436580 $D/p_gp1 \
  > $D/p_gp1.log 2>&1
log "p_gp1 rc $?"
log "queue done"
