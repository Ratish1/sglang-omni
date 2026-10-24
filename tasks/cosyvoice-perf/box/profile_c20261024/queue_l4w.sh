#!/usr/bin/env bash
# 10-24, cards 0 and 1, after queue_l4p.sh: the worst step of #3129's head (t_l4p) at 16 rows, the vocoder's default
# max_batch_size, for the PR's peak memory line, and main's at 16 rows beside it on card 1.
# Decision rule, written before the run: L4 passes when every W16 and c64 request completes with no allocator failure,
# no allocator retry and no traceback.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261024/l4p/queue_w.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261024/queue_l4w.sh'
D=/data/c20261024
P=$D/l4p
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
until grep -q 'queue l4p done' $P/queue.txt; do sleep 30; done
[ -s $D/worst16/meta.lst ] || (cd /workspace/t_mn && PYTHONPATH=/workspace/t_mn python3 \
  $IN/make_worst_meta_c20261011.py --out $D/worst16 --requests 16 > $P/worst_meta_16.log 2>&1)
[ -s $D/worst16/meta.lst ] || { log "worst list failed, stop"; exit 1; }
python3 $IN/worst_step_c20261021.py /workspace/t_l4p 0 $(cpus 0) 46000 $P/worst_t_l4p_16 16 $D/worst16/meta.lst > $P/worst_t_l4p_16.txt 2>&1 &
python3 $IN/worst_step_c20261021.py /workspace/t_mn 1 $(cpus 1) 46400 $P/worst_t_mn_16 16 $D/worst16/meta.lst > $P/worst_t_mn_16.txt 2>&1 &
wait
for name in t_l4p t_mn; do
  log "worst step $name 16 rows: $(grep -E 'W[0-9]+:|c64:' $P/worst_${name}_16.txt | tr -s ' ' | tr '\n' ' ' | cut -c1-360)"
  grep -E 'alloc retries|outside torch|reserve|Flow prefix cache' $P/worst_${name}_16.txt | sed 's/^/    /' >> $P/queue.txt
done
log "queue l4w done"
