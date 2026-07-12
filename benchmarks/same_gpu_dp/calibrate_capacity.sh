#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Discover a reliable equal KV cap from repeated capacity-only launches.
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

: "${CALIBRATION_DPS:=2,3,4}"
: "${CALIBRATION_REPETITIONS:=2}"
: "${CALIBRATION_MPS_MODES:=0,1}"
: "${CALIBRATION_MARGIN_BPS:=0}"
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
[[ "$CALIBRATION_MARGIN_BPS" =~ ^[0-9]+$ ]] && \
  ((CALIBRATION_MARGIN_BPS < 10000)) || {
  echo "CALIBRATION_MARGIN_BPS must be an integer in [0, 9999]" >&2
  exit 2
}
if [[ -e "$CALIBRATION_ROOT" ]]; then
  echo "refusing to overwrite calibration directory: $CALIBRATION_ROOT" >&2
  exit 2
fi
mkdir -p "$CALIBRATION_ROOT"

ENV_FILE="$CALIBRATION_ROOT/capacity.env"
SUMMARY_FILE="$CALIBRATION_ROOT/capacity_summary.tsv"
SELECTION_FILE="$CALIBRATION_ROOT/capacity_selection.tsv"
: > "$ENV_FILE"
printf 'dp\tmps\trepetition\trotation\tminimum_tokens\tkv_capacity_json\n' > "$SUMMARY_FILE"
printf 'dp\tprofiled_min_tokens\tmargin_basis_points\tsafe_cap_tokens\n' > "$SELECTION_FILE"

rotate_list() {
  local raw=$1 separator=$2 shift=$3
  local parts=() result=() i count joined=""
  IFS="$separator" read -r -a parts <<< "$raw"
  count=${#parts[@]}
  for ((i = 0; i < count; i++)); do
    result+=("${parts[$(((i + shift) % count))]}")
  done
  for ((i = 0; i < count; i++)); do
    [[ -n "$joined" ]] && joined+="$separator"
    joined+="${result[$i]}"
  done
  printf '%s\n' "$joined"
}

IFS=',' read -r -a MPS_MODES <<< "$CALIBRATION_MPS_MODES"
for mps in "${MPS_MODES[@]}"; do
  [[ "$mps" == 0 || "$mps" == 1 ]] || {
    echo "CALIBRATION_MPS_MODES entries must be 0 or 1; got '$mps'" >&2
    exit 2
  }
done

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
  for mps in "${MPS_MODES[@]}"; do
    for ((rep = 1; rep <= CALIBRATION_REPETITIONS; rep++)); do
      rotation=$(((rep - 1) % dp))
      server_sets=$(rotate_list "${!server_var}" ";" "$rotation")
      client_sets=$(rotate_list "${!client_var}" ";" "$rotation")
      mem_fractions=$(rotate_list "${!mem_var}" "," "$rotation")
      label="calibrate_dp${dp}_mps${mps}_rep${rep}"
      LABEL="$label" DP="$dp" USE_MPS="$mps" MODE=direct \
        SERVER_CORE_SETS="$server_sets" CLIENT_CORE_SETS="$client_sets" \
        MEM_FRACTIONS="$mem_fractions" MAX_TOTAL_TOKENS= CAPACITY_ONLY=1 \
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
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$dp" "$mps" "$rep" "$rotation" "$condition_min" "$kv_path" >> "$SUMMARY_FILE"
      if [[ -z "$dp_min" || "$condition_min" -lt "$dp_min" ]]; then
        dp_min=$condition_min
      fi
    done
  done
  safe_cap=$((dp_min * (10000 - CALIBRATION_MARGIN_BPS) / 10000))
  printf 'export DP%s_PROFILED_MIN_TOKENS=%s\n' "$dp" "$dp_min" >> "$ENV_FILE"
  printf 'export DP%s_MAX_TOTAL_TOKENS=%s\n' "$dp" "$safe_cap" >> "$ENV_FILE"
  printf '%s\t%s\t%s\t%s\n' \
    "$dp" "$dp_min" "$CALIBRATION_MARGIN_BPS" "$safe_cap" >> "$SELECTION_FILE"
done

echo "capacity calibration complete: $CALIBRATION_ROOT"
echo "review $SUMMARY_FILE and $SELECTION_FILE, then run: source $ENV_FILE"
