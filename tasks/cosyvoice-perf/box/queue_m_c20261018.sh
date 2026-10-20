#!/usr/bin/env bash
# 10-18, cards 2 and 3: A/A on main (t_ic, 369776998's tree, in both slots), every cell today's pairs ran (c1 seeded,
# c1u, c16, c64, r16), both orders, so p_ev1, p_ev2, p_ev12, p_la, p_st and p_stm read their tails against today's
# box. The tree is warm from queue_i.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_m.pgid; exec bash queue_m_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_m.txt; }

log "queue start"
(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261018.py /workspace/t_ic \
  /workspace/t_ic $D/aa_m > $D/aa_m.log 2>&1)
log "pair aa_m rc $?"
log "queue done"
