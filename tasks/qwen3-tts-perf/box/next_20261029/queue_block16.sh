#!/usr/bin/env bash
# Block 16 of 10-29, after block 15b (WAIT_PID): where the compiled reference encode's saving lands. Block 15b's pairs
# put the speedups branch at -1.1 and -0.8 % RTF non-streaming c16 but level streaming (+0.3, -0.4 %), while block 11's
# float32 encode, 0.67 ms more device time, cost +1.06 % streaming. Profile pairs as block 11b (block 8's nsys flags,
# 640 requests at c16, both card orders, streaming then non-streaming), main against /workspace/n29_compile, read by
# encode_overlap.py: encode window, talker step time beside an encode and without one.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
S=$T/tasks/omni_step_profiling_20260918/scripts
OVERLAP=$T/tasks/qwen3-tts-perf/box/next_20261028/encode_overlap.py
W=$D/block16
L=$W/queue.log
mkdir -p $W/profile
if [ -n "${WAIT_PID:-}" ]; then
  while [ -n "$(ps -o stat= -p $WAIT_PID 2> /dev/null | grep -v Z)" ]; do sleep 30; done
else
  :
fi
echo "queue block16 start $(date +%T) kit $(git -C $T rev-parse HEAD) compile $(git -C /workspace/n29_compile rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L

profile() {
  local tree=$1 out=$2 card=$3 port=$4 stream_flag=$5
  BENCH_STREAM=$stream_flag PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh /workspace/n29_$tree $out $card $port 16 640 > $out.out 2>&1
  echo "profile $(basename $out) rc $? $(date +%T)" >> $L
}

for mode in stream nonstream; do
  if [ $mode = stream ]; then flag=--stream; else flag=""; fi
  profile main $W/profile/${mode}_a_main 0 26200 "$flag" &
  profile compile $W/profile/${mode}_a_compile 1 26700 "$flag" &
  wait
  profile compile $W/profile/${mode}_b_compile 0 27200 "$flag" &
  profile main $W/profile/${mode}_b_main 1 27700 "$flag" &
  wait
done
python3 $OVERLAP $W/profile/*_a_*/ $W/profile/*_b_*/ > $W/overlap.txt 2>&1
echo "overlap rc $? $(date +%T)" >> $L
echo "queue block16 done $(date +%T)" >> $L
