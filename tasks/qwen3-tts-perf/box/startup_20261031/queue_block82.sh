#!/usr/bin/env bash
# Block 82 of 11-02: why non-streaming throughput stays at 44 req/s from concurrency 64 to 256 (block 80) while streaming
# reaches 70 to 86. nsys with the pipeline probe and the interpreter-lock trace on #3418's final head, card 0, SeedTTS EN,
# non-streaming then streaming at concurrency 64, 640 timed requests each; pipeline_census.py per run.
# usage: D=... T=<kit tree> PR=<tree> bash queue_block82.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
PR=${PR:?the tree}
S=$T/tasks/omni_step_profiling_20260918/scripts
W=$D/block82
L=$W/queue.log
mkdir -p $W
echo "tree $PR $(git -C $PR rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "queue block82 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block79/boot_pr/inductor TRITON_CACHE_DIR=$D/block79/boot_pr/triton
PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py BENCH_STREAM="" \
  NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
  bash $S/run_nsys_boot.sh $PR $W/nostream 0 23000 64 640 > $W/nostream.out 2>&1
echo "nostream rc $? $(date +%T)" >> $L
PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
  NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
  bash $S/run_nsys_boot.sh $PR $W/stream 0 23400 64 640 > $W/stream.out 2>&1
echo "stream rc $? $(date +%T)" >> $L
echo "queue block82 done $(date +%T)" >> $L
