#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1, after queue s2: an A/A of the stack's head t_l4 against itself at c1f,
# c16 and c64, both orders, the noise band at the loads where 10-22 has none (every A/A of the day was c16 or c16p),
# with every second's host load (both slots' cores, the other node, every card) from host_load_c20261022.py; then the
# pair audit's rows and each cell's load.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_aa.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_aa_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $D/queue_aa.txt; }
until grep -q "queue s2 done" $D/queue_s2.txt; do sleep 30; done
log "queue aa start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) t_l4 $(git -C /workspace/t_l4 log --oneline -1 | cut -c1-10) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
python3 $IN/host_load_c20261022.py $D/p_aa1_load.jsonl &
sampler=$!
(cd /workspace/sglang-omni && PAIR_CELLS=c1f,c16,c64 python3 $IN/serve_pair_c20261021.py /workspace/t_l4 /workspace/t_l4 \
  $D/p_aa1 > $D/p_aa1.log 2>&1)
log "pair p_aa1 rc $? token cap hits $(grep -E '\| at the token cap \|' $D/p_aa1.log | awk -F'|' '{printf "%s:%s/%s ", $3, $5, $6}')"
kill $sampler
for cell in c1f c16 c64; do
  python3 $IN/pooled_inter_chunk_c20261021.py $D/p_aa1 $cell t_l4 t_l4 > $D/pooled_p_aa1_$cell.txt 2>&1
  log "pooled p_aa1 $cell $(grep '^pooled' $D/pooled_p_aa1_$cell.txt | tr '\n' ' ')"
done
python3 $IN/host_load_c20261022.py --summary $D/p_aa1_load.jsonl $D/p_aa1 c1f,c16,c64 > $D/p_aa1_load.txt 2>&1
python3 $IN/pair_audit_c20261022.py $D /workspace/sglang-omni > $D/pair_audit_aa.txt 2>&1
log "audit $(grep '^| p_aa1 ' $D/pair_audit_aa.txt | cut -c1-400)"
log "queue aa done"
