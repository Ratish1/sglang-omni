#!/usr/bin/env bash
# 10-20, cards 2 and 3: the AR loop and the vocoder on disjoint SM partitions (the probe's COSY_GREEN_AR_SMS switch,
# green_streams.py) in the real pipeline. The candidate with P1 (t_cp20b, d27c42294) profiled at the same time,
# plain on card 2 and with 32 SMs for the AR on card 3 (c16, c64): the AR's decode step alone and shared, every
# section's device time, each request's states. Scripts from the trace marks checkout.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_e.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_e_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
S=t_cp20b
R=$D/prof_e
log() { echo "$* $(date +%T)" >> $D/queue_e.txt; }

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) tree $S sglang_omni $(git -C /workspace/$S rev-parse HEAD:sglang_omni | cut -c1-8)"
mkdir -p $R
PEER=$R/green PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261020.sh $S 2 0-15,64-79 27600 \
  $R/plain > $R/run_plain.log 2>&1 &
PEER=$R/plain COSY_GREEN_AR_SMS=32 PYSPY=0 CELLS="c16:16:320 c64:64:640" bash $IN/cosy_profile_run_c20261020.sh $S 3 \
  16-31,80-95 27800 $R/green > $R/run_green.log 2>&1 &
wait
log "profiles $(tail -n 1 $R/plain/progress.txt) / $(tail -n 1 $R/green/progress.txt)"
log "green lines: $(grep -h cosy_green $R/green/serve.log | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
for cell in c16 c64; do
  python3 $IN/request_states_c20261019.py plain=$R/plain/$cell.sqlite green=$R/green/$cell.sqlite > $R/states_$cell.txt 2>&1
done
log "queue done"
