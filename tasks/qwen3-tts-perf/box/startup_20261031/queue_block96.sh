#!/usr/bin/env bash
# Block 96 of 11-02: the streaming baseline of the 10x program on its chain base, card 0, cards 1 to 3 idle, one nsys at a
# time. SM Issue's known answers on this card, then SeedTTS EN 1.7B Base streaming at concurrency 1 (320 requests) and 64
# (640): nsys with the gh100 metrics, CUDA and NVTX at graph level, the pipeline probe; window means, by owner, the census.
# Then concurrency 1 with the interpreter-lock trace for host states on the first-audio and per-frame paths. Every serve
# log is read for a probe target that no longer patches.
# usage: D=... T=<kit tree> BASE=<tree> bash queue_block96.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
BASE=${BASE:?the baseline tree}
S=$T/tasks/omni_step_profiling_20260918/scripts
W=$D/block96
L=$W/queue.log
mkdir -p $W/known
echo "tree BASE $BASE $(git -C $BASE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $BASE diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue block96 start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $W/known && nsys profile -o $W/known/known --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=0 --gpu-metrics-set=gh100 python3 $S/sm_issue_known_answers.py > known.log 2>&1 \
  && nsys export --type sqlite --force-overwrite=true -o $W/known/known.sqlite $W/known/known.nsys-rep > export.log 2>&1 \
  && python3 $S/known_answer_ranges.py $W/known/known.sqlite > $W/known/known_answers.txt 2>&1)
echo "known rc $? $(date +%T)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor TRITON_CACHE_DIR=$W/cache/triton
METRICS="--trace=cuda,nvtx --cuda-graph-trace=graph --sample=none --cpuctxsw=none --gpu-metrics-devices=0 --gpu-metrics-set=gh100"
LOCK="--trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=graph --sample=none --cpuctxsw=none"
port=20000
run() {
  local label=$1 concurrency=$2 samples=$3 args=$4
  local out=$W/$label
  PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py NSYS_ARGS="$args" \
    bash $S/run_nsys_boot.sh $BASE $out 0 $port $concurrency $samples > $out.out 2>&1
  echo "$label rc $? $(date +%T); probe patch failures $(grep -c 'pipeline_nvtx patch' $out/serve.log)" >> $L
  port=$((port + 400))
}
run warm_compile 1 8 "--trace=cuda --sample=none --cpuctxsw=none"
run metrics_c1 1 320 "$METRICS"
run metrics_c64 64 640 "$METRICS"
for label in metrics_c1 metrics_c64; do
  (cd $S && python3 gpu_metrics_window.py $W/$label/serve.sqlite --bench-log $W/$label/bench.log > $W/$label/gpu_metrics.txt 2>&1 \
    && python3 metrics_by_owner.py $W/$label/serve.sqlite --bench-log $W/$label/bench.log > $W/$label/by_owner.txt 2>&1)
  echo "$label reads rc $? $(date +%T)" >> $L
done
run lock_c1 1 320 "$LOCK"
echo "queue block96 done $(date +%T)" >> $L
