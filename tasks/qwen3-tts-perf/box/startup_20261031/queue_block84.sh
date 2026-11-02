#!/usr/bin/env bash
# Block 84 of 11-02: where non-streaming c64 goes once whole requests decode on the codec state. Waits for block 83. nsys
# with the pipeline probe and the interpreter-lock trace, card 0, SeedTTS EN, non-streaming at concurrency 64, 640 timed
# requests; pipeline_census.py. Read against block 82's nostream census of the base.
# usage: D=... T=<kit tree> WHOLE=<tree> bash queue_block84.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
WHOLE=${WHOLE:?the tree}
S=$T/tasks/omni_step_profiling_20260918/scripts
W=$D/block84
L=$W/queue.log
mkdir -p $W
until grep -q "queue block83 done" $D/block83/queue.log; do sleep 5; done
echo "tree $WHOLE $(git -C $WHOLE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "queue block84 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block83/cache_whole/inductor TRITON_CACHE_DIR=$D/block83/cache_whole/triton
PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py BENCH_STREAM="" \
  NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
  bash $S/run_nsys_boot.sh $WHOLE $W/nostream 0 24000 64 640 > $W/nostream.out 2>&1
echo "nostream rc $? $(date +%T)" >> $L
echo "queue block84 done $(date +%T)" >> $L
