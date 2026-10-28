#!/usr/bin/env bash
# Block 8 of 10-28: where PR D's c16 cost lands. Main (upstream 1f0834e9c) and D (f14c83813, the same main merged in)
# served under nsys with the pipeline probe and the interpreter-lock trace, graphs traced whole, SeedTTS EN 1.7B Base,
# 640 requests at c16; each run ends in the pipeline census (A to H) over the benchmark's window. Same-time pairs on
# cards 0 and 1, then swapped; streaming first, then non-streaming.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
S=$T/tasks/omni_step_profiling_20260918/scripts
MAIN=/workspace/n28u_main
REFENC=/workspace/n28u_refenc
W=$D/profile_d
L=$W/queue.log
mkdir -p $W
echo "queue block8 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')refenc $(git -C $REFENC rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L

profile() {
  local tree=$1 out=$2 card=$3 port=$4 stream_flag=$5
  BENCH_STREAM=$stream_flag PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh $tree $out $card $port 16 640 > $out.out 2>&1
  echo "$(basename $out) rc $? $(date +%T)" >> $L
}

for mode in stream nonstream; do
  if [ $mode = stream ]; then flag=--stream; else flag=""; fi
  profile $MAIN $W/${mode}_main_a 0 31000 "$flag" &
  profile $REFENC $W/${mode}_refenc_a 1 31500 "$flag" &
  wait
  profile $REFENC $W/${mode}_refenc_b 0 32000 "$flag" &
  profile $MAIN $W/${mode}_main_b 1 32500 "$flag" &
  wait
done
echo "queue block8 done $(date +%T)" >> $L
