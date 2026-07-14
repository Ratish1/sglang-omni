#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run a caller-specified DP x MPS x concurrency matrix with fixed CPU budgets.
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
DRY_RUN=0
DRY_RUN_ARG=()
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  DRY_RUN_ARG=(--dry-run)
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

: "${MATRIX_ORDER:=1:0,1:1,2:0,2:1,3:0,3:1,4:0,4:1}"
: "${CONCURRENCY_VALUES:=32,48,64,96}"
: "${REPETITIONS:=1}"
: "${SHUFFLE_SEED:=1}"
: "${RUN_LABEL:=matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
: "${OUT_ROOT:=$REPO/benchmarks/results/same_gpu_dp/$RUN_LABEL}"

[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || {
  echo "REPETITIONS must be a positive integer" >&2
  exit 2
}
[[ "$SHUFFLE_SEED" == off || "$SHUFFLE_SEED" =~ ^[0-9]+$ ]] || {
  echo "SHUFFLE_SEED must be a non-negative integer or 'off'" >&2
  exit 2
}
[[ "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "RUN_LABEL must contain only letters, numbers, dot, underscore, and dash" >&2
  exit 2
}
if [[ -e "$OUT_ROOT" ]]; then
  echo "refusing to overwrite existing matrix directory: $OUT_ROOT" >&2
  exit 2
fi

IFS=',' read -r -a CONCURRENCIES <<< "$CONCURRENCY_VALUES"
for concurrency in "${CONCURRENCIES[@]}"; do
  [[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid CONCURRENCY_VALUES item '$concurrency'" >&2
    exit 2
  }
done

server_budget=""
client_budget=""
IFS=',' read -r -a CONDITIONS <<< "$MATRIX_ORDER"
for condition in "${CONDITIONS[@]}"; do
  IFS=':' read -r dp mps extra <<< "$condition"
  [[ -z "${extra:-}" && "$dp" =~ ^[1-6]$ && "$mps" =~ ^[01]$ ]] || {
    echo "invalid MATRIX_ORDER item '$condition' (expected DP:MPS)" >&2
    exit 2
  }
  server_var="DP${dp}_SERVER_CORE_SETS"
  client_var="DP${dp}_CLIENT_CORE_SETS"
  mem_var="DP${dp}_MEM_FRACTIONS"
  token_var="DP${dp}_MAX_TOTAL_TOKENS"
  [[ -n "${!server_var:-}" && -n "${!client_var:-}" && -n "${!mem_var:-}" ]] || {
    echo "set $server_var, $client_var, and $mem_var" >&2
    exit 2
  }
  if [[ "${KV_EQUALITY:-warn}" == require && "$dp" -gt 1 && -z "${!token_var:-}" ]]; then
    echo "KV_EQUALITY=require needs $token_var for DP$dp; run a capacity-only calibration first" >&2
    exit 2
  fi
  layout=$(python3 "$HERE/summarize.py" validate-layout --dp "$dp" \
    --server-core-sets "${!server_var}" --client-core-sets "${!client_var}")
  counts=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["server_cpu_count"], d["client_cpu_count"])' <<< "$layout")
  read -r current_server_budget current_client_budget <<< "$counts"
  [[ -z "$server_budget" ]] && server_budget=$current_server_budget
  [[ -z "$client_budget" ]] && client_budget=$current_client_budget
  if [[ "$current_server_budget" -ne "$server_budget" || "$current_client_budget" -ne "$client_budget" ]]; then
    echo "DP$dp CPU budget ($current_server_budget server, $current_client_budget client) differs from matrix budget ($server_budget, $client_budget)" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT"
RESULTS_TSV="$OUT_ROOT/matrix_results.tsv"
printf 'repetition\torder_index\tdp\tmps\tconcurrency\tstatus\texit_code\toutput_dir\n' > "$RESULTS_TSV"
{
  echo "run_label=$RUN_LABEL"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
  echo "matrix_order=$MATRIX_ORDER"
  echo "concurrency_values=$CONCURRENCY_VALUES"
  echo "repetitions=$REPETITIONS"
  echo "shuffle_seed=$SHUFFLE_SEED"
  echo "server_cpu_budget=$server_budget"
  echo "client_cpu_budget=$client_budget"
  env | LC_ALL=C sort | grep -E '^DP[1-6]_(SERVER_CORE_SETS|CLIENT_CORE_SETS|MEM_FRACTIONS|MAX_TOTAL_TOKENS|MPS_THREAD_PERCENTAGES|MPS_PINNED_MEM_LIMITS)=' || true
} > "$OUT_ROOT/matrix_manifest.txt"

failures=0
cells=""
for condition in "${CONDITIONS[@]}"; do
  for concurrency in "${CONCURRENCIES[@]}"; do
    [[ -n "$cells" ]] && cells+=","
    cells+="${condition}:${concurrency}"
  done
done
for ((rep = 1; rep <= REPETITIONS; rep++)); do
  rep_cells=$(python3 - "$cells" "$SHUFFLE_SEED" "$rep" <<'PY'
import random
import sys

cells = sys.argv[1].split(",")
if sys.argv[2] != "off":
    random.Random(int(sys.argv[2]) + int(sys.argv[3])).shuffle(cells)
print(",".join(cells))
PY
)
  IFS=',' read -r -a REP_CELLS <<< "$rep_cells"
  order_index=0
  for cell in "${REP_CELLS[@]}"; do
    order_index=$((order_index + 1))
    IFS=':' read -r dp mps concurrency <<< "$cell"
    server_var="DP${dp}_SERVER_CORE_SETS"
    client_var="DP${dp}_CLIENT_CORE_SETS"
    mem_var="DP${dp}_MEM_FRACTIONS"
    token_var="DP${dp}_MAX_TOTAL_TOKENS"
    thread_var="DP${dp}_MPS_THREAD_PERCENTAGES"
    pinned_var="DP${dp}_MPS_PINNED_MEM_LIMITS"
    label="rep${rep}_ord${order_index}_dp${dp}_mps${mps}_c${concurrency}_direct"
    status=pass
    thread_value=""
    pinned_value=""
    if [[ "$mps" -eq 1 ]]; then
      thread_value=${!thread_var:-}
      pinned_value=${!pinned_var:-}
    fi
    if LABEL="$label" DP="$dp" USE_MPS="$mps" MODE=direct \
      SERVER_CORE_SETS="${!server_var}" CLIENT_CORE_SETS="${!client_var}" \
      MEM_FRACTIONS="${!mem_var}" CONCURRENCY_PER_WORKER="$concurrency" \
      MAX_TOTAL_TOKENS="${!token_var:-${MAX_TOTAL_TOKENS:-}}" \
      MPS_THREAD_PERCENTAGES="$thread_value" \
      MPS_PINNED_MEM_LIMITS="$pinned_value" \
      OUT_ROOT="$OUT_ROOT" "$HERE/run_condition.sh" "${DRY_RUN_ARG[@]}"; then
      exit_code=0
    else
      exit_code=$?
    fi
    if [[ "$exit_code" -eq 130 || "$exit_code" -eq 143 ]]; then
      echo "matrix interrupted while running $label" >&2
      exit "$exit_code"
    elif [[ "$exit_code" -ne 0 ]]; then
      status=fail
      failures=$((failures + 1))
    elif [[ "$DRY_RUN" -eq 1 ]]; then
      status=dry-run
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$rep" "$order_index" "$dp" "$mps" "$concurrency" "$status" "$exit_code" \
      "$OUT_ROOT/$label" >> "$RESULTS_TSV"
  done
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  python3 "$HERE/summarize.py" summarize-matrix \
    --output "$OUT_ROOT/matrix_summary.json" "$RESULTS_TSV"
fi

if [[ "$failures" -gt 0 ]]; then
  echo "matrix completed with $failures failed conditions; see $RESULTS_TSV" >&2
  exit 1
fi
echo "matrix complete: $OUT_ROOT"
