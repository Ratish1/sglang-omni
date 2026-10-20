#!/usr/bin/env bash
# 10-20, card 2, after queue g. The CUDA speech tokenizer alone per session mode (tokenizer_modes_c20261020.py) from
# main's tree: why prerequisite 1's run lock costs c16 first audio on cache misses (p_q1), and which mode keeps the
# arena bounded without serializing host bound runs.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_h.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_h_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
log() { echo "$* $(date +%T)" >> $D/queue_h.txt; }

until grep -q "queue done" $D/queue_g.txt 2> /dev/null; do sleep 30; done
log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
(cd /workspace/t_m20 && CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/t_m20 taskset -c 0-15,64-79 \
  python3 $IN/tokenizer_modes_c20261020.py $D/tokenizer_modes > $D/tokenizer_modes.txt 2>&1)
log "tokenizer modes rc $? $(grep -c tokens_equal_main $D/tokenizer_modes.txt) modes"
log "queue done"
