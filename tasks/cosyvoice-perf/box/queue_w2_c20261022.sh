#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: the worst step checks again with the fixed memory probe (it imported torch
# on a thread in inductor's compile workers, whose pool then exited: queue m and n's runs void). t_l4 (L4 with L3) and
# t_l24 (L4 without L3: ONNX Runtime's tokenizer, whose arena grows while serving) at 16 rows on one card each, then at
# 32 rows; each W<rows> (identical seeded requests at the text limit) then stream c64, with peak use, memory outside
# torch, allocation retries and out of memory failures per process.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_w2.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_w2_c20261022.sh'
D=/data/c20261022b
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $D/queue_w2.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
log "queue w2 start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
for rows in 16 32; do
  worst=$D/worst/meta.lst
  [ $rows = 32 ] && worst=$D/worst32/meta.lst
  python3 $IN/worst_step_c20261021.py /workspace/t_l4 1 $(cpus 1) 46000 $D/w2_l4_$rows $rows $worst > $D/w2_l4_$rows.txt 2>&1 &
  python3 $IN/worst_step_c20261021.py /workspace/t_l24 0 $(cpus 0) 46400 $D/w2_l24_$rows $rows $worst > $D/w2_l24_$rows.txt 2>&1 &
  wait
  for arm in l4 l24; do
    log "worst step $arm $rows rows: $(grep -E 'W[0-9]+:|c64:' $D/w2_${arm}_$rows.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-360)"
    grep -E 'alloc retries|outside torch|peak' $D/w2_${arm}_$rows.txt | sed 's/^/    /' >> $D/queue_w2.txt
    log "  compile pool exits $(grep -c 'SubprocPool unclean exit' $D/w2_${arm}_$rows/serve.log)"
  done
done
log "queue w2 done"
