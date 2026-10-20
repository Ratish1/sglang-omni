#!/usr/bin/env bash
# 10-20 evening: the SM Issue baseline of main and the vocoder's device time by function, from the profiling branch
# (main bcd0d8c68 plus its instrumentation), card 0 with card 1 idle, one boot at a time. GPU metrics sample card 0
# only (the container sees the other task's cards, and the readers do not filter by device).
#  1. known answers (SM_ISSUE_20261006.md 1): a saturated matmul, an HBM-bound add, the talker's 64-row matmul;
#  2. c16 and c64, 320 requests: nsys with the gh100 metrics, CUDA and NVTX at graph level, the trace marks probe;
#     window means, by what the device runs, the pipeline census;
#  3. c16, 320 requests: node-level graph trace with capture-time line labels; the census G section and the vocoder's
#     device time by function (V3, V4, V5 and the conv glue at today's sizes, HISTORY_20261007.md 9).
set -u
D=/data/sm_issue_20261020
T=${T:-/workspace/t_marks}
S=$T/tasks/omni_step_profiling_20260918/scripts
PROBE=$S/pipeline_nvtx
L=$D/queue.log
mkdir -p $D/known
echo "queue sm start $(date +%T) marks $(git -C $T rev-parse HEAD) tree $(git -C $T rev-parse HEAD:sglang_omni)" >> $L
(cd $D/known && nsys profile -o $D/known/known --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=0 --gpu-metrics-set=gh100 python3 $S/sm_issue_known_answers.py > known.log 2>&1 \
  && nsys export --type sqlite --force-overwrite=true -o $D/known/known.sqlite $D/known/known.nsys-rep > export.log 2>&1 \
  && python3 $S/known_answer_ranges.py $D/known/known.sqlite > $D/known/known_answers.txt 2>&1)
echo "known rc $? $(date +%T)" >> $L
PORT=11000
for C in 16 64; do
  OUT=$D/metrics_c$C
  NSYS_ARGS="--trace=cuda,nvtx --cuda-graph-trace=graph --sample=none --cpuctxsw=none --gpu-metrics-devices=0 --gpu-metrics-set=gh100" \
    PROBE_PATH=$PROBE PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py \
    bash $S/run_nsys_boot.sh $T $OUT 0 $PORT $C 320 > $OUT.out 2>&1
  echo "metrics_c$C rc $? $(date +%T)" >> $L
  (cd $S && python3 gpu_metrics_window.py $OUT/serve.sqlite --bench-log $OUT/bench.log > $OUT/gpu_metrics.txt 2>&1 \
    && python3 metrics_by_owner.py $OUT/serve.sqlite --bench-log $OUT/bench.log > $OUT/by_owner.txt 2>&1)
  echo "metrics_c$C reads rc $? $(date +%T)" >> $L
  PORT=$((PORT + 1000))
done
OUT=$D/lines_c16
NSYS_ARGS="--trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none" \
  PROBE_PATH=$PROBE PROBE_ENV="OMNI_PIPE_NVTX=1 OMNI_PIPE_LINES=capture" CENSUS=pipeline_census.py \
  bash $S/run_nsys_boot.sh $T $OUT 0 $PORT 16 320 > $OUT.out 2>&1
echo "lines_c16 rc $? $(date +%T)" >> $L
(cd $S && python3 pipeline_census.py $OUT/serve.sqlite --bench-log $OUT/bench.log --sections G --top 2000 \
  > $OUT/census_Gall.txt 2>&1 && python3 vocoder_function_census.py $OUT/census_Gall.txt > $OUT/vocoder_functions.txt 2>&1)
echo "lines_c16 reads rc $? $(date +%T)" >> $L
echo queue_sm_done $(date +%T) >> $L
