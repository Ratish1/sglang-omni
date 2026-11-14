#!/usr/bin/env bash
# step_bench.sh <head> <card>: on the host, the predictor codebook-step bench of one head (whose
# worktree run_head.sh made), once no other session holds a GPU.
HEAD=$1
CARD=$2
C=sglang-omni-qwen3-tts-predictor-20261114
B=/data/qwen3-tts-synthetic-20261114
OURS="persistent_layer_check|persistent_stamps|persistent_ncu|predictor_bench|latency_probe|small_gemm_probe"
foreign() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    ps -o cmd= -p $p | grep -qE "$OURS" || echo $p
  done
}
until [ -z "$(foreign)" ]; do sleep 15; done
echo "$(date +%H:%M:%S) start step bench $HEAD card $CARD"
podman exec $C bash -c "cd $B/kit/tasks/qwen3_tts_synthetic_profile_20261114 && CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$B/heads/$HEAD timeout 900 python3 predictor_bench.py --out $B/predictor_bench/h200_${HEAD}_step.json > $B/predictor_bench/h200_${HEAD}_step.log 2>&1"
echo "$(date +%H:%M:%S) done step bench $HEAD exit $?"
