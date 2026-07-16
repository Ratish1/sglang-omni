#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Find the largest equal KV cap that every same-GPU replica can resolve.
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
: "${CONDITION_RUNNER:=$HERE/run_condition.sh}"

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi

: "${CALIBRATION_DPS:=2,3,4}"
: "${CALIBRATION_MPS_MODES:=1}"
: "${CALIBRATION_CONFIRMATIONS:=3}"
: "${CALIBRATION_TOKEN_TOLERANCE:=256}"
: "${CALIBRATION_GROWTH_FACTOR:=2}"
: "${CALIBRATION_MAX_SEARCH_TRIALS:=20}"
: "${CALIBRATION_MARGIN_BPS:=0}"
: "${CALIBRATION_LABEL:=capacity_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${CALIBRATION_ROOT:=$REPO/benchmarks/results/same_gpu_dp/$CALIBRATION_LABEL}"

[[ "$CALIBRATION_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "CALIBRATION_LABEL contains unsupported characters" >&2
  exit 2
}
for name in CALIBRATION_CONFIRMATIONS CALIBRATION_TOKEN_TOLERANCE \
  CALIBRATION_MAX_SEARCH_TRIALS; do
  value=${!name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$name must be a positive integer" >&2
    exit 2
  }
done
[[ "$CALIBRATION_GROWTH_FACTOR" =~ ^[2-9][0-9]*$ ]] || {
  echo "CALIBRATION_GROWTH_FACTOR must be an integer >= 2" >&2
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
TRIALS_FILE="$CALIBRATION_ROOT/capacity_trials.tsv"
SELECTION_FILE="$CALIBRATION_ROOT/capacity_selection.tsv"
: > "$ENV_FILE"
printf 'dp\tphase\ttrial\tmps\trotation\tcandidate_tokens\tstatus\tresolved_tokens\toutput_dir\n' \
  > "$TRIALS_FILE"
printf 'dp\tmem_fractions\thighest_passing_tokens\tfirst_failing_tokens\ttolerance_tokens\tmargin_basis_points\tselected_tokens\n' \
  > "$SELECTION_FILE"

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

classify_kv() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
try:
    values = list(json.load(open(path, encoding="utf-8")).values())
except (OSError, ValueError) as exc:
    print(f"invalid:{exc}")
    raise SystemExit(20)
if not values or any(value is None for value in values):
    print("invalid:missing")
    raise SystemExit(20)
resolved = [int(value) for value in values]
print(",".join(str(value) for value in resolved))
raise SystemExit(0 if all(value == expected for value in resolved) else 10)
PY
}

startup_oom_log() {
  local out=$1 log
  for log in "$out"/workers/*.server.log; do
    [[ -f "$log" ]] || continue
    if grep -Fq "torch.OutOfMemoryError: CUDA out of memory." "$log"; then
      printf '%s\n' "$log"
      return 0
    fi
  done
  return 1
}

IFS=',' read -r -a MPS_MODES <<< "$CALIBRATION_MPS_MODES"
for mps in "${MPS_MODES[@]}"; do
  [[ "$mps" == 0 || "$mps" == 1 ]] || {
    echo "CALIBRATION_MPS_MODES entries must be 0 or 1; got '$mps'" >&2
    exit 2
  }
done

trial_id=0
run_candidate() {
  local dp=$1 mps=$2 cap=$3 phase=$4 rotation=$5
  local server_var="DP${dp}_SERVER_CORE_SETS"
  local client_var="DP${dp}_CLIENT_CORE_SETS"
  local mem_var="DP${dp}_MEM_FRACTIONS"
  local server_sets client_sets mem_fractions label out kv_path
  local run_status classify_status resolved status oom_log

  trial_id=$((trial_id + 1))
  server_sets=$(rotate_list "${!server_var}" ";" "$rotation")
  client_sets=$(rotate_list "${!client_var}" ";" "$rotation")
  mem_fractions=$(rotate_list "${!mem_var}" "," "$rotation")
  label="capacity_dp${dp}_${phase}${trial_id}_mps${mps}_cap${cap}"
  out="$CALIBRATION_ROOT/$label"

  if LABEL="$label" DP="$dp" USE_MPS="$mps" MODE=direct \
    SERVER_CORE_SETS="$server_sets" CLIENT_CORE_SETS="$client_sets" \
    MEM_FRACTIONS="$mem_fractions" MAX_TOTAL_TOKENS="$cap" CAPACITY_ONLY=1 \
    KV_EQUALITY=require MPS_THREAD_PERCENTAGES= MPS_PINNED_MEM_LIMITS= \
    OUT_ROOT="$CALIBRATION_ROOT" "$CONDITION_RUNNER"; then
    run_status=0
  else
    run_status=$?
  fi
  if [[ "$run_status" -eq 130 || "$run_status" -eq 143 ]]; then
    return "$run_status"
  fi

  kv_path="$out/kv_capacity.json"
  if [[ ! -f "$kv_path" ]]; then
    if [[ "$run_status" -ne 0 ]] && oom_log=$(startup_oom_log "$out"); then
      printf '%s\t%s\t%s\t%s\t%s\t%s\tcapacity-limit\tstartup-oom\t%s\n' \
        "$dp" "$phase" "$trial_id" "$mps" "$rotation" "$cap" "$out" \
        >> "$TRIALS_FILE"
      echo "candidate $cap exceeded startup memory capacity; see $oom_log" >&2
      return 1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\tinfrastructure-failure\tmissing\t%s\n' \
      "$dp" "$phase" "$trial_id" "$mps" "$rotation" "$cap" "$out" \
      >> "$TRIALS_FILE"
    echo "candidate $cap failed before producing KV evidence; inspect $out" >&2
    return 2
  fi
  if resolved=$(classify_kv "$kv_path" "$cap"); then
    classify_status=0
  else
    classify_status=$?
  fi

  if [[ "$run_status" -eq 0 && "$classify_status" -eq 0 ]]; then
    status=pass
  elif [[ "$classify_status" -eq 10 ]]; then
    status=capacity-limit
  else
    status=infrastructure-failure
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$dp" "$phase" "$trial_id" "$mps" "$rotation" "$cap" "$status" \
    "$resolved" "$out" >> "$TRIALS_FILE"

  [[ "$status" == pass ]] && return 0
  [[ "$status" == capacity-limit ]] && return 1
  echo "candidate $cap failed for a non-capacity reason; inspect $out" >&2
  return 2
}

evaluate_candidate() {
  local dp=$1 cap=$2 phase=$3 rotation=$4 mps result
  for mps in "${MPS_MODES[@]}"; do
    if run_candidate "$dp" "$mps" "$cap" "$phase" "$rotation"; then
      continue
    else
      result=$?
      return "$result"
    fi
  done
}

IFS=',' read -r -a DPS <<< "$CALIBRATION_DPS"
for dp in "${DPS[@]}"; do
  if [[ ! "$dp" =~ ^[1-9][0-9]*$ ]] || ((dp < 2 || dp > 10)); then
    echo "CALIBRATION_DPS entries must be between 2 and 10; got '$dp'" >&2
    exit 2
  fi
  server_var="DP${dp}_SERVER_CORE_SETS"
  client_var="DP${dp}_CLIENT_CORE_SETS"
  mem_var="DP${dp}_MEM_FRACTIONS"
  initial_var="DP${dp}_INITIAL_CAP_TOKENS"
  [[ -n "${!server_var:-}" && -n "${!client_var:-}" && \
     -n "${!mem_var:-}" && -n "${!initial_var:-}" ]] || {
    echo "set $server_var, $client_var, $mem_var, and $initial_var" >&2
    exit 2
  }
  [[ "${!initial_var}" =~ ^[1-9][0-9]*$ ]] || {
    echo "$initial_var must be a positive integer" >&2
    exit 2
  }

  search_trials=0
  low=${!initial_var}
  if evaluate_candidate "$dp" "$low" seed 0; then
    :
  else
    result=$?
    [[ "$result" -eq 1 ]] \
      && echo "known-safe seed $low no longer fits DP$dp" >&2 \
      || echo "DP$dp seed trial failed for a non-capacity reason" >&2
    exit 1
  fi
  search_trials=$((search_trials + 1))

  while :; do
    ((search_trials < CALIBRATION_MAX_SEARCH_TRIALS)) || {
      echo "DP$dp exceeded CALIBRATION_MAX_SEARCH_TRIALS while bracketing" >&2
      exit 1
    }
    high=$((low * CALIBRATION_GROWTH_FACTOR))
    if evaluate_candidate "$dp" "$high" bracket "$((search_trials % dp))"; then
      low=$high
      search_trials=$((search_trials + 1))
      continue
    else
      result=$?
      [[ "$result" -eq 1 ]] || exit "$result"
      search_trials=$((search_trials + 1))
      break
    fi
  done

  while ((high - low > CALIBRATION_TOKEN_TOLERANCE)); do
    ((search_trials < CALIBRATION_MAX_SEARCH_TRIALS)) || {
      echo "DP$dp exceeded CALIBRATION_MAX_SEARCH_TRIALS during binary search" >&2
      exit 1
    }
    mid=$(((low + high) / 2))
    if evaluate_candidate "$dp" "$mid" search "$((search_trials % dp))"; then
      low=$mid
    else
      result=$?
      [[ "$result" -eq 1 ]] || exit "$result"
      high=$mid
    fi
    search_trials=$((search_trials + 1))
  done

  selected=$((low * (10000 - CALIBRATION_MARGIN_BPS) / 10000))
  for ((confirmation = 1; confirmation <= CALIBRATION_CONFIRMATIONS; confirmation++)); do
    if ! evaluate_candidate "$dp" "$selected" confirm "$(((confirmation - 1) % dp))"; then
      echo "DP$dp selected cap $selected failed confirmation $confirmation" >&2
      exit 1
    fi
  done

  printf 'export DP%s_HIGHEST_PASS_TOKENS=%s\n' "$dp" "$low" >> "$ENV_FILE"
  printf 'export DP%s_MAX_TOTAL_TOKENS=%s\n' "$dp" "$selected" >> "$ENV_FILE"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$dp" "${!mem_var}" "$low" "$high" "$CALIBRATION_TOKEN_TOLERANCE" \
    "$CALIBRATION_MARGIN_BPS" "$selected" >> "$SELECTION_FILE"
done

echo "equal-capacity search complete: $CALIBRATION_ROOT"
echo "review $TRIALS_FILE and $SELECTION_FILE, then run: source $ENV_FILE"
