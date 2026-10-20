#!/usr/bin/env bash
# 10-20, Qwen3-TTS Base profile (PLAN.md 3, host pass): nsys with the pipeline probe and the interpreter-lock trace,
# graphs traced whole, from the profiling branch 827c4b4af (tree 1e8dea03b), one boot per cell on card 0, after the
# mapping queue. Each run ends in the pipeline census (sections A to H) over the benchmark's window.
set -u
D=/data/prof_20261020
T=${T:-/workspace/t_marks}
S=$T/tasks/omni_step_profiling_20260918/scripts
L=$D/queue.log
while pgrep -f "^bash queue_profile_2.sh" > /dev/null; do sleep 10; done
echo "queue nsys start $(date +%T)" >> $L
for cell in "c64 64 320" "c16 16 320" "c1 1 60"; do
  set -- $cell
  PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh $T $D/nsys_$1 0 32000 $2 $3 > $D/nsys_$1.out 2>&1
  echo "nsys_$1 rc $? $(date +%T)" >> $L
done
echo queue_nsys_done $(date +%T) >> $L
