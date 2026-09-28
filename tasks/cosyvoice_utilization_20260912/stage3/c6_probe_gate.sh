#!/usr/bin/env bash
# Probe gate of a candidate against main, one card, one run at a time: startup cold and warm,
# then the one call probe with compile on and with compile off, for each head.
#   NEW=<sha9> MAIN=<sha9> OUT=/data/gate bash c6_probe_gate.sh > /data/gate/x.log 2>&1
set -u
export CUDA_VISIBLE_DEVICES=${CARD:-0} OMP_NUM_THREADS=1
T=/workspace/sglang-omni/.tmp/wt/analysis/tasks/cosyvoice_utilization_20260912/stage3
W=/workspace/sglang-omni/.tmp/wt
mkdir -p "$OUT"
for head in "$NEW" "$MAIN"; do
  cd "$W/c3_$head"
  rm -rf "$OUT/inductor_$head"
  for run in cold warm; do
    echo "=== startup $head $run $(date +%T)"
    TORCHINDUCTOR_CACHE_DIR="$OUT/inductor_$head" TORCH_LOGS="recompiles,graph_breaks" \
      PYTHONPATH="$PWD" python3 "$T/c1_compile_startup.py" --out "$OUT/startup_${head}_$run.json" \
      > "$OUT/startup_${head}_$run.log" 2>&1
    echo "rc=$? $(date +%T)"
  done
done
for mode in compile eager; do
  flag=""
  if [ "$mode" = eager ]; then flag="--no-compile"; else :; fi
  for head in "$NEW" "$MAIN"; do
    cd "$W/c3_$head"
    echo "=== hop $head $mode $(date +%T)"
    TORCHINDUCTOR_CACHE_DIR="$OUT/inductor_$head" PYTHONPATH="$PWD" \
      python3 "$T/c2_hop_cost.py" $flag --out "$OUT/hop_${head}_$mode.json" \
      > "$OUT/hop_${head}_$mode.log" 2>&1
    echo "rc=$? $(date +%T)"
  done
done
echo ALL_DONE
