#!/usr/bin/env bash
# Block 103 of 11-02: profile under the serving stress load, long streaming requests (4,096 characters, 2,048 frames).
# nsys with the metrics, the pipeline probe and the census, c64 with 128 requests and c1 with 3, one nsys at a time on one
# card. Then one SeedTTS c64 census of the same tree, for the same-head comparison.
# usage: D=... T=<kit tree> TREE=<tree> HARNESS=<tree> BLOCK=block103 CARD=2 PORT=30000 bash queue_block103.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
TREE=${TREE:?the tree}
HARNESS=${HARNESS:?the harness tree}
BLOCK=${BLOCK:-block103}
CARD=${CARD:-2}
port=${PORT:-30000}
K=$T/tasks/qwen3-tts-perf/box/startup_20261031
S=$T/tasks/omni_step_profiling_20260918/scripts
W=$D/$BLOCK
L=$W/queue.log
mkdir -p $W
echo "tree $TREE $(git -C $TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $TREE diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
echo "queue $BLOCK start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export TORCHINDUCTOR_CACHE_DIR=$D/block96/cache/inductor TRITON_CACHE_DIR=$D/block96/cache/triton
METRICS="--trace=cuda,nvtx --cuda-graph-trace=graph --sample=none --cpuctxsw=none --gpu-metrics-devices=$CARD --gpu-metrics-set=gh100"
reads() {
  local label=$1
  (cd $S && python3 gpu_metrics_window.py $W/$label/serve.sqlite --bench-log $W/$label/bench.log > $W/$label/gpu_metrics.txt 2>&1 \
    && python3 metrics_by_owner.py $W/$label/serve.sqlite --bench-log $W/$label/bench.log > $W/$label/by_owner.txt 2>&1)
  echo "$label reads rc $? $(date +%T)" >> $L
}
stress() {
  local label=$1 concurrency=$2 requests=$3 args=${4:-$METRICS}
  local out=$W/$label
  PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py NSYS_ARGS="$args" \
    SERVE_ARGS="--allowed-local-media-path $HARNESS/docs/_static/audio" LOAD_SCRIPT=$K/stress_load.py \
    LOAD_ARGS="--harness $HARNESS --spec $HARNESS/benchmarks/tts_serving/examples/stress_qwen3_tts_long_c64.json" \
    bash $S/run_nsys_boot.sh $TREE $out $CARD $port $concurrency $requests > $out.out 2>&1
  echo "$label rc $? $(date +%T); probe patch failures $(grep -c 'pipeline_nvtx patch' $out/serve.log); $(tail -n 1 $out/bench.log | cut -c1-120)" >> $L
  port=$((port + 400))
  reads $label
}
stress long_c64 64 128
stress long_c1 1 3
out=$W/seed_c64
PROBE_PATH=$S/pipeline_nvtx PROBE_ENV="OMNI_PIPE_NVTX=1" CENSUS=pipeline_census.py NSYS_ARGS="$METRICS" \
  bash $S/run_nsys_boot.sh $TREE $out $CARD $port 64 640 > $out.out 2>&1
echo "seed_c64 rc $? $(date +%T)" >> $L
reads seed_c64
echo "queue $BLOCK done $(date +%T)" >> $L
