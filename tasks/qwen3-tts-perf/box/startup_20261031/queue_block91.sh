#!/usr/bin/env bash
# Block 91 of 11-02: who holds the interpreter lock at concurrency 64 on the whole request head, streaming on card 0 and
# non-streaming on card 1, and the non-streaming census of where its device and host time now go. Waits for block 90.
# usage: D=... T=<kit tree> WHOLE=<tree> bash queue_block91.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
WHOLE=${WHOLE:?the tree}
K=$T/tasks/qwen3-tts-perf/box
S=$T/tasks/omni_step_profiling_20260918/scripts
W=$D/block91
L=$W/queue.log
mkdir -p $W
until grep -q "queue block90 done" $D/block90/queue.log 2> /dev/null; do sleep 5; done
echo "tree $WHOLE $(git -C $WHOLE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
echo "queue block91 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $K/startup_20261031 && python3 gil_owners.py --tree $WHOLE --root $W/gil_stream --card 0 --port 25000 > $W/gil_stream.txt 2>&1) &
(cd $K/startup_20261031 && python3 gil_owners.py --tree $WHOLE --root $W/gil_nostream --card 1 --port 25400 --no-stream > $W/gil_nostream.txt 2>&1) &
wait
for mode in stream nostream; do
  echo "gil $mode: $(grep -E '^lock samples|^[a-zA-Z_N-]+: [0-9.]+ %' $W/gil_$mode.txt | head -n 8 | tr '\n' ';' | cut -c1-300)" >> $L
done
PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py BENCH_STREAM="" \
  NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
  bash $S/run_nsys_boot.sh $WHOLE $W/nostream 0 25800 64 640 > $W/nostream.out 2>&1
echo "nostream census rc $? $(date +%T)" >> $L
echo "queue block91 done $(date +%T)" >> $L
