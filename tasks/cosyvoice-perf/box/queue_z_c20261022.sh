#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: a third c16 pair of t_l2 against t_l3, both orders, gaps pooled. p_r3 (the
# old node) read L3's pooled gap mean +2.5 %, p_y2 (this node) -1.0 %, the A/A p_y1 +0.5 % pooled with card 1 slower
# than card 0 by 0.9 to 1.8 % inside an order.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_z.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_z_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_z.txt; }
log "queue z start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
(cd /workspace/sglang-omni && PAIR_CELLS=c16 python3 $IN/serve_pair_c20261021.py /workspace/t_l2 /workspace/t_l3 $D/p_y3 \
  > $D/p_y3.log 2>&1)
log "pair p_y3 rc $? token cap hits $(grep -E '\| at the token cap \|' $D/p_y3.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
python3 $IN/pooled_inter_chunk_c20261021.py $D/p_y3 c16 t_l2 t_l3 > $D/pooled_p_y3_c16.txt 2>&1
log "pooled p_y3 c16 $(grep '^pooled' $D/pooled_p_y3_c16.txt | tr '\n' ' ')"
log "queue z done"
