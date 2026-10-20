#!/usr/bin/env bash
# 10-20, card 3 alone, after queue c: whether CUDA green contexts isolate a decode-like graph from a persistent GEMM
# load on one H100 (green_context_probe_c20261020.py, DESIGN_PIPELINE_SECTIONS_20261020.md 7). Scripts from the
# trace marks checkout, updated from origin first.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261020/queue_d.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/queue_d_c20261020.sh'
D=/data/c20261020
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
log() { echo "$* $(date +%T)" >> $D/queue_d.txt; }

log "queue start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD)"
mkdir -p $D/probes
CUDA_VISIBLE_DEVICES=3 taskset -c 16-31,80-95 python3 $IN/green_context_probe_c20261020.py --replays 400 \
  > $D/probes/green_context.txt 2>&1
log "green context probe rc $? $(grep -c . $D/probes/green_context.txt) lines"
log "queue done"
