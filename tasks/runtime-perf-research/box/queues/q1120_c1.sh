#!/usr/bin/env bash
# 11-20, node-0000: the c1 lone video stage time, main (serial, OMP 112, decode 16) against the rework (8 video
# requests, decode 16), m l l m on card 0's socket after both GPU queues finished, nothing else running.
set -u
O=/data/v1120/cpu
K=/workspace/kit/tasks/runtime-perf-research/box
log() { echo "$* $(date -u +%T)" >> /data/v1120/queue.log; }
until [ "$(grep -c 'q1120_gpu done' /data/v1120/queue.log)" -ge 2 ]; do sleep 30; done
log "q1120_c1 start load $(cut -d' ' -f1-3 /proc/loadavg)"
for arm in main vlane vlane main; do
  index=$(ls $O | grep -c "^c1q_$arm.*json")
  if [ $arm = main ]; then config=1:112:16; else config=8:112:16; fi
  PYTHONPATH=/workspace/$arm OMP_NUM_THREADS=112 taskset -c 0-55,112-167 python3 $K/preprocess_c1_profile.py \
    --configs $config --out $O/c1q_${arm}_$index.json > $O/c1q_${arm}_$index.log 2>&1
  log "c1q $arm $index rc $? $(tail -1 $O/c1q_${arm}_$index.log)"
done
log "q1120_c1 done"
