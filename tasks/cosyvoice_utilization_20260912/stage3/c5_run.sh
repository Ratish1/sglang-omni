#!/usr/bin/env bash
# Startup probe (cold, warm) of one head, then the hop cost probe on each head.
#   HEAD_NEW=<sha9> bash c5_run.sh > /data/c5/x.log 2>&1
set -u
export CUDA_VISIBLE_DEVICES=${CARD:-0} OMP_NUM_THREADS=1
T=/workspace/sglang-omni/.tmp/wt/analysis/tasks/cosyvoice_utilization_20260912/stage3
W=/workspace/sglang-omni/.tmp/wt
OUT=${OUT:-/data/c5}
mkdir -p "$OUT"
cd "$W/c3_$HEAD_NEW"
for run in cold warm; do
  echo "=== new startup $run $(date +%T)"
  if [ "$run" = cold ]; then rm -rf "$OUT/inductor_new"; else :; fi
  TORCHINDUCTOR_CACHE_DIR="$OUT/inductor_new" TORCH_LOGS="recompiles,graph_breaks" \
    PYTHONPATH="$PWD" python3 "$T/c1_compile_startup.py" --out "$OUT/new_$run.json" \
    > "$OUT/new_$run.log" 2>&1
  echo "rc=$? $(date +%T)"
done
for pair in new:$HEAD_NEW $HOP_OTHERS; do
  name=${pair%%:*}; rev=${pair##*:}
  cd "$W/c3_$rev"
  echo "=== hop $name $(date +%T)"
  TORCHINDUCTOR_CACHE_DIR="$OUT/inductor_$name" PYTHONPATH="$PWD" \
    python3 "$T/c2_hop_cost.py" --out "$OUT/hop_$name.json" > "$OUT/hop_$name.log" 2>&1
  echo "rc=$? $(date +%T)"
done
echo ALL_DONE
