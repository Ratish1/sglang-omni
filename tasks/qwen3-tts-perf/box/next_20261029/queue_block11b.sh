#!/usr/bin/env bash
# Block 11b of 10-29: block 11's profile half, trimmed to what block 8 has not answered. Block 8 (main 1f0834e9c against
# D f14c83813, the same encode path as today's D) already gives D's beside-encode cost and the observable's card band
# (main a / b), so only the two halves are profiled: fp32 and exact against main, both card orders, streaming then
# non-streaming (block 8's nsys flags, 640 requests at c16), read by encode_overlap.py. Waits for block 11's served round
# 2, then stops block 11 (BLOCK11_PGID) before its own profile loop runs.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
S=$T/tasks/omni_step_profiling_20260918/scripts
OVERLAP=$T/tasks/qwen3-tts-perf/box/next_20261028/encode_overlap.py
L=$D/queue.log
until grep -q "served round 2 done" $L; do sleep 5; done
kill -TERM -- -${BLOCK11_PGID:?block 11 queue pgid} 2> /dev/null
for second in $(seq 180); do pgrep -g $BLOCK11_PGID > /dev/null || break; sleep 1; done
pkill -KILL -g $BLOCK11_PGID 2> /dev/null
for second in $(seq 180); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$used" -lt 1024 ] && break
  sleep 1
done
echo "cards free to $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')$(date +%T)" >> $L
rm -rf $D/profile
mkdir -p $D/profile
echo "queue block11b start $(date +%T) kit $(git -C $T rev-parse HEAD), block 11 stopped after its served pairs" >> $L

profile() {
  local tree=$1 out=$2 card=$3 port=$4 stream_flag=$5
  BENCH_STREAM=$stream_flag PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    NSYS_ARGS="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none" \
    bash $S/run_nsys_boot.sh /workspace/n29_$tree $out $card $port 16 640 > $out.out 2>&1
  echo "profile $(basename $out) rc $? $(date +%T)" >> $L
}

for mode in stream nonstream; do
  if [ $mode = stream ]; then flag=--stream; else flag=""; fi
  for x in fp32 exact; do
    profile main $D/profile/${mode}_a_main_vs_$x 0 26200 "$flag" &
    profile $x $D/profile/${mode}_a_$x 1 26700 "$flag" &
    wait
    profile $x $D/profile/${mode}_b_$x 0 27200 "$flag" &
    profile main $D/profile/${mode}_b_main_vs_$x 1 27700 "$flag" &
    wait
  done
done
python3 $OVERLAP $D/profile/*_a_*/ $D/profile/*_b_*/ > $D/overlap.txt 2>&1
echo "overlap rc $? $(date +%T)" >> $L
echo "queue block11b done $(date +%T)" >> $L
