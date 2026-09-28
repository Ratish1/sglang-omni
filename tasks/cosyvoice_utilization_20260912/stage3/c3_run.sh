#!/usr/bin/env bash
# Native DiT compile variants from one tree, each cold then warm on its own cache.
#   REV=<sha9> CARD=0 OUT=/data/c8 bash c3_run.sh > /data/c8/x.log 2>&1
set -u
export CUDA_VISIBLE_DEVICES=${CARD:-0} OMP_NUM_THREADS=1
T=/workspace/sglang-omni/.tmp/wt/analysis/tasks/cosyvoice_utilization_20260912/stage3
cd "/workspace/sglang-omni/.tmp/wt/c3_$REV"
mkdir -p "$OUT"
for variant in eager main nested nested_disable; do
  rm -rf "$OUT/inductor_$variant"
  for run in cold warm; do
    echo "=== $variant $run $(date +%T)"
    TORCHINDUCTOR_CACHE_DIR="$OUT/inductor_$variant" TORCH_LOGS="recompiles" \
      PYTHONPATH="$PWD" python3 "$T/c3_native_compile.py" --variant "$variant" \
      --out "$OUT/$run" > "$OUT/${variant}_$run.log" 2>&1
    echo "rc=$? $(date +%T)"
  done
done
echo ALL_DONE
