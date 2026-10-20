#!/usr/bin/env bash
# 10-18, cards 2 and 3: the r16 tail of p_ship (order a first audio p99 +104 %, one 0.9 s process wide stall at request
# 918 on card 3; p_st and p_stdl2 had one like it in order a on card 3, never on card 2 or in order b, never in main's
# arms). Two more rounds of main (t_ic) against the ship stack (t_ship), r16 only, both orders, so a recurrence is
# told from a one off. Waits for queue_r.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_s.pgid; exec bash queue_s_c20261018.sh'
D=/data/c20261018
IN=$D/in
log() { echo "$* $(date +%T)" >> $D/queue_s.txt; }

log "queue wait"
until grep -q "queue done" $D/queue_r.txt; do sleep 30; done
log "queue start"
for round in p_ship_r1 p_ship_r2; do
  (cd /workspace/sglang-omni && PAIR_CELLS=r16 python3 $IN/serve_pair_c20261018.py /workspace/t_ic /workspace/t_ship \
    $D/$round > $D/$round.log 2>&1)
  log "pair $round rc $?"
done
log "queue done"
