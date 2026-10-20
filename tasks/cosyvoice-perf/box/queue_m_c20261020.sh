#!/usr/bin/env bash
# 10-20, card 3, after queue l: the Euler solve census for L7 again on t_l6c (queue l's run sized the prefix pool at
# 4 GB, short of 8 rows of the longest hop on both twins).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_m.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_m_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
log() { echo "$* $(date +%T)" >> $D/queue_m.txt; }

until grep -q "queue done" $D/queue_l.txt 2> /dev/null; do sleep 30; done
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
(cd /workspace/t_l6c && CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/t_l6c:$COSYVOICE TORCHINDUCTOR_CACHE_DIR=$D/inductor_t_l6c \
  taskset -c 16-31,80-95 python3 $IN/solve_census_c20261020.py $D/solve_census.txt > $D/solve_census.log 2>&1)
log "census rc $? $(grep -c kernels $D/solve_census.txt) steps"
log "queue done"
