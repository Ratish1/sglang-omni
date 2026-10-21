#!/usr/bin/env bash
# 10-22, the 0.5.22 container, cards 0 and 1, after queue u: L4's worst step check on this base (0.5.22, #3029's
# HiFT): t_l4 at 16 rows on card 1 and at 32 rows on card 0, each with W<rows> (the 10-21 worst lists, identical
# seeded requests at the text limit) then stream c64, the memory probe on; peak use, memory outside torch, out of
# memory retries and failures.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022r/queue_v.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_v_c20261022.sh'
D=/data/c20261022r
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_r.txt; }
until grep -q "queue u done" $D/queue_r.txt; do sleep 30; done
log "queue v start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
python3 $IN/worst_step_c20261021.py /workspace/t_l4 1 16-31,80-95 46000 $D/worst_l4_16 16 /data/c20261021/worst/meta.lst \
  > $D/worst_l4_16.txt 2>&1 &
python3 $IN/worst_step_c20261021.py /workspace/t_l4 0 0-15,64-79 46400 $D/worst_l4_32 32 /data/c20261021/worst32/meta.lst \
  > $D/worst_l4_32.txt 2>&1 &
wait
for rows in 16 32; do
  log "worst step $rows rows: $(grep -E 'W[0-9]+:|c64:' $D/worst_l4_$rows.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-330)"
  grep 'alloc retries' $D/worst_l4_$rows.txt | sed 's/^/    /' >> $D/queue_r.txt
done
log "queue v done"
