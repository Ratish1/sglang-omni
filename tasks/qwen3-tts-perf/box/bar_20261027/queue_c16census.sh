#!/usr/bin/env bash
# The streaming c16 cost of the no-wait branch, named: main then the branch served under Nsight Systems with the
# pipeline_nvtx probe and the interpreter-lock trace, SeedTTS EN 1.7B Base streaming at c16 (400 samples), the census
# over the benchmark's timed window (all sections: D is the device overlap, E the lock). Card 0.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
S=$T/tasks/omni_step_profiling_20260918/scripts
mkdir -p $D
for arm in main:/workspace/main_72b776c9c nowait:/workspace/stack_50b590e2d; do
  label=${arm%%:*}
  tree=${arm#*:}
  echo "$label start $(date +%T) tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $D/log.txt
  PROBE_PATH=$S/pipeline_nvtx PROBE_ENV=OMNI_PIPE_NVTX=1 CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh $tree $D/$label 0 36000 16 400 > $D/$label.out 2>&1
  echo "$label rc $? $(date +%T)" >> $D/log.txt
done
echo "c16census_done $(date +%T)" >> $D/log.txt
