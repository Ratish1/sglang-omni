#!/usr/bin/env bash
# 10-18, cards 2 and 3: two more A/A rounds on main (t_ic in both slots), c64 only, both orders: the c64 inter chunk
# p95 and p99 tails of fix 1 (+7.5, +9.6 % pooled) sit near today's and 10-17's A/A spread (+3.3, -2.1, -7.1 % at p95),
# and three more samples of that spread decide whether they are past it.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_n.pgid; exec bash queue_n_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_n.txt; }

log "queue start"
for round in aa_n1 aa_n2; do
  (cd /workspace/sglang-omni && PAIR_CELLS=c64 python3 $IN/serve_pair_c20261018.py /workspace/t_ic /workspace/t_ic \
    $D/$round > $D/$round.log 2>&1)
  log "pair $round rc $?"
done
log "queue done"
