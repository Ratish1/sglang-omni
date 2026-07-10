#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Discover the smallest profiled KV capacity for each DP across MPS off/on.
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

: "${CALIBRATION_DPS:=2,3,4}"
: "${CALIBRATION_REPETITIONS:=2}"
: "${CALIBRATION_LABEL:=capacity_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${CALIBRATION_ROOT:=$REPO/benchmarks/results/same_gpu_dp/$CALIBRATION_LABEL}"

[[ "$CALIBRATION_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "CALIBRATION_LABEL contains unsupported characters" >&2
  exit 2
}
[[ "$CALIBRATION_REPETITIONS" =~ ^[1-9][0-9]*$ ]] || {
  echo "CALIBRATION_REPETITIONS must be a positive integer" >&2
  exit 2
}
if [[ -e "$CALIBRATION_ROOT" ]]; then
  echo "refusing to overwrite calibration directory: $CALIBRATION_ROOT" >&2
  exit 2
fi
mkdir -p "$CALIBRATION_ROOT"

ENV_FILE="$CALIBRATION_ROOT/capacity.env"
SUMMARY_FILE="$CALIBRATION_ROOT/capacity_summary.tsv"
: > "$ENV_FILE"
printf 'dp\tmps\trepetition\tminimum_tokens\tkv_capacity_json\n' > "$SUMMARY_FILE"

IFS=',' read -r -a DPS <<< "$CALIBRATION_DPS"
for dp in "${DPS[@]}"; do
  [[ "$dp" =~ ^[2-4]$ ]] || {
    echo "CALIBRATION_DPS entries must be 2, 3, or 4; got '$dp'" >&2
    exit 2
  }
  server_var="DP${dp}_SERVER_CORE_SETS"
  client_var="DP${dp}_CLIENT_CORE_SETS"
  mem_var="DP${dp}_MEM_FRACTIONS"
  [[ -n "${!server_var:-}" && -n "${!client_var:-}" && -n "${!mem_var:-}" ]] || {
    echo "set $server_var, $client_var, and $mem_var" >&2
    exit 2
  }

  dp_min=""
  for mps in 0 1; do
    for ((rep = 1; rep <= CALIBRATION_REPETITIONS; rep++)); do
      label="calibrate_dp${dp}_mps${mps}_rep${rep}"
      LABEL="$label" DP="$dp" USE_MPS="$mps" MODE=direct \
        SERVER_CORE_SETS="${!server_var}" CLIENT_CORE_SETS="${!client_var}" \
        MEM_FRACTIONS="${!mem_var}" MAX_TOTAL_TOKENS= CAPACITY_ONLY=1 \
        KV_EQUALITY=warn MPS_THREAD_PERCENTAGES= MPS_PINNED_MEM_LIMITS= \
        OUT_ROOT="$CALIBRATION_ROOT" "$HERE/run_condition.sh"

      kv_path="$CALIBRATION_ROOT/$label/kv_capacity.json"
      condition_min=$(python3 - "$kv_path" <<'PY'
import json
import sys

values = list(json.load(open(sys.argv[1], encoding="utf-8")).values())
if not values or any(value is None for value in values):
    raise SystemExit(f"missing KV capacity in {sys.argv[1]}")
print(min(int(value) for value in values))
PY
      )
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$dp" "$mps" "$rep" "$condition_min" "$kv_path" >> "$SUMMARY_FILE"
      if [[ -z "$dp_min" || "$condition_min" -lt "$dp_min" ]]; then
        dp_min=$condition_min
      fi
    done
  done
  printf 'export DP%s_MAX_TOTAL_TOKENS=%s\n' "$dp" "$dp_min" >> "$ENV_FILE"
done

echo "capacity calibration complete: $CALIBRATION_ROOT"
echo "review $SUMMARY_FILE, then run: source $ENV_FILE"
