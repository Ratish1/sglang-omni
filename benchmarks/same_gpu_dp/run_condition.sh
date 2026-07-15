#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch and benchmark one same-GPU DP condition. Configuration is via the
# documented environment variables in README.md; pass --dry-run to validate
# placement and print commands without touching CUDA, MPS, or model servers.
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

: "${MODEL:=boson-sglang/higgs-audio-v3-tts-4b-base}"
: "${MODEL_NAME:=higgs}"
: "${META:=zhaochenyang20/seed-tts-eval-arrow}"
: "${CLIENT_DRIVER:=seedtts}"
: "${TTS_MANIFEST:=}"
: "${DP:=1}"
: "${USE_MPS:=0}"
: "${MODE:=direct}"
: "${GPU_UUID:=}"
: "${NUMA_NODE:=0}"
: "${SERVER_CORE_SETS:=0-15}"
: "${CLIENT_CORE_SETS:=16-23}"
: "${ROUTER_CORES:=}"
: "${ROUTER_POLICY:=least_request}"
: "${BASE_PORT:=8801}"
: "${ROUTER_PORT:=8799}"
: "${MEM_FRACTIONS:=auto}"
: "${CONCURRENCY_PER_WORKER:=16}"
: "${MAX_RUNNING_REQUESTS:=64}"
: "${CUDA_GRAPH_MAX_BS:=64}"
: "${MAX_TOTAL_TOKENS:=}"
: "${MAX_NEW_TOKENS:=2048}"
: "${MAX_SAMPLES:=}"
: "${SEED:=1}"
: "${WARMUP:=1}"
: "${BENCH_LANG:=en}"
: "${REF_FORMAT:=references}"
: "${ALLOWED_LOCAL_MEDIA_PATH:=/}"
: "${STREAM:=0}"
: "${SERVER_READY_TIMEOUT:=1200}"
: "${MPS_READY_TIMEOUT:=30}"
: "${SHUTDOWN_TIMEOUT:=90}"
: "${CLIENT_READY_TIMEOUT:=600}"
: "${KV_EQUALITY:=warn}"
: "${CAPACITY_ONLY:=0}"
: "${KEEP_AUDIO:=0}"
: "${REQUIRE_IDLE_GPU:=1}"
: "${MPS_THREAD_PERCENTAGES:=}"
: "${MPS_PINNED_MEM_LIMITS:=}"
: "${OUT_ROOT:=$REPO/benchmarks/results/same_gpu_dp}"
: "${MPS_TMP_ROOT:=/tmp/sglang-omni-mps-${UID:-$(id -u)}}"
: "${LABEL:=dp${DP}_mps${USE_MPS}_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! "$DP" =~ ^[1-9][0-9]*$ ]] || ((DP > 10)); then
  echo "DP must be between 1 and 10" >&2
  exit 2
fi
[[ "$USE_MPS" =~ ^[01]$ ]] || { echo "USE_MPS must be 0 or 1" >&2; exit 2; }
[[ "$STREAM" =~ ^[01]$ ]] || { echo "STREAM must be 0 or 1" >&2; exit 2; }
[[ "$REQUIRE_IDLE_GPU" =~ ^[01]$ ]] || {
  echo "REQUIRE_IDLE_GPU must be 0 or 1" >&2
  exit 2
}
[[ "$CAPACITY_ONLY" =~ ^[01]$ ]] || {
  echo "CAPACITY_ONLY must be 0 or 1" >&2
  exit 2
}
[[ "$KEEP_AUDIO" =~ ^[01]$ ]] || {
  echo "KEEP_AUDIO must be 0 or 1" >&2
  exit 2
}
[[ "$NUMA_NODE" =~ ^[0-9]+$ ]] || { echo "NUMA_NODE must be non-negative" >&2; exit 2; }
[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "LABEL must contain only letters, numbers, dot, underscore, and dash" >&2
  exit 2
}
[[ "$BENCH_LANG" == en || "$BENCH_LANG" == zh ]] || {
  echo "BENCH_LANG must be en or zh" >&2
  exit 2
}
[[ "$REF_FORMAT" == flat || "$REF_FORMAT" == references ]] || {
  echo "REF_FORMAT must be flat or references" >&2
  exit 2
}
for numeric_name in BASE_PORT ROUTER_PORT CONCURRENCY_PER_WORKER MAX_RUNNING_REQUESTS \
  CUDA_GRAPH_MAX_BS MAX_NEW_TOKENS SERVER_READY_TIMEOUT SHUTDOWN_TIMEOUT \
  CLIENT_READY_TIMEOUT; do
  numeric_value=${!numeric_name}
  [[ "$numeric_value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$numeric_name must be a positive integer" >&2
    exit 2
  }
done
[[ "$MPS_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "MPS_READY_TIMEOUT must be a positive integer" >&2
  exit 2
}
if [[ -n "$MAX_TOTAL_TOKENS" && ! "$MAX_TOTAL_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOTAL_TOKENS must be empty or a positive integer" >&2
  exit 2
fi
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "WARMUP must be a non-negative integer" >&2; exit 2; }
if [[ -n "$MAX_SAMPLES" && ! "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_SAMPLES must be empty or a positive integer" >&2
  exit 2
fi
((BASE_PORT + DP - 1 <= 65535 && ROUTER_PORT <= 65535)) || {
  echo "worker/router port exceeds 65535" >&2
  exit 2
}
if [[ "$USE_MPS" -eq 0 && (-n "$MPS_THREAD_PERCENTAGES" || -n "$MPS_PINNED_MEM_LIMITS") ]]; then
  echo "MPS resource settings require USE_MPS=1" >&2
  exit 2
fi
[[ "$MODE" == direct || "$MODE" == router ]] || {
  echo "MODE must be direct or router" >&2
  exit 2
}
[[ "$ROUTER_POLICY" == round_robin || "$ROUTER_POLICY" == least_request || \
  "$ROUTER_POLICY" == random ]] || {
  echo "ROUTER_POLICY must be round_robin, least_request, or random" >&2
  exit 2
}
[[ "$CLIENT_DRIVER" == seedtts || "$CLIENT_DRIVER" == manifest ]] || {
  echo "CLIENT_DRIVER must be seedtts or manifest" >&2
  exit 2
}
if [[ "$CLIENT_DRIVER" == manifest ]]; then
  [[ -n "$TTS_MANIFEST" && -f "$TTS_MANIFEST" ]] || {
    echo "TTS_MANIFEST must name a readable JSONL file for CLIENT_DRIVER=manifest" >&2
    exit 2
  }
  TTS_MANIFEST=$(cd "$(dirname "$TTS_MANIFEST")" && pwd)/$(basename "$TTS_MANIFEST")
fi
[[ "$KV_EQUALITY" == off || "$KV_EQUALITY" == warn || "$KV_EQUALITY" == require ]] || {
  echo "KV_EQUALITY must be off, warn, or require" >&2
  exit 2
}
if [[ "$MODE" == router && -z "$ROUTER_CORES" ]]; then
  echo "ROUTER_CORES is required in router mode" >&2
  exit 2
fi
if [[ "$CAPACITY_ONLY" -eq 1 && "$MODE" != direct ]]; then
  echo "CAPACITY_ONLY=1 requires MODE=direct" >&2
  exit 2
fi

IFS=';' read -r -a SERVER_CORES <<< "$SERVER_CORE_SETS"
IFS=';' read -r -a CLIENT_CORES <<< "$CLIENT_CORE_SETS"
IFS=',' read -r -a MEM_FRAC <<< "$MEM_FRACTIONS"
IFS=',' read -r -a MPS_THREAD_PCT <<< "$MPS_THREAD_PERCENTAGES"
IFS=';' read -r -a MPS_PINNED_LIMIT <<< "$MPS_PINNED_MEM_LIMITS"

if [[ ${#MEM_FRAC[@]} -eq 1 && $DP -gt 1 ]]; then
  first=${MEM_FRAC[0]}
  MEM_FRAC=()
  for ((i = 0; i < DP; i++)); do MEM_FRAC+=("$first"); done
fi
[[ ${#MEM_FRAC[@]} -eq $DP ]] || {
  echo "MEM_FRACTIONS must contain one value or exactly DP=$DP values" >&2
  exit 2
}
if [[ -n "$MPS_THREAD_PERCENTAGES" ]]; then
  if [[ ${#MPS_THREAD_PCT[@]} -eq 1 && $DP -gt 1 ]]; then
    first=${MPS_THREAD_PCT[0]}
    MPS_THREAD_PCT=()
    for ((i = 0; i < DP; i++)); do MPS_THREAD_PCT+=("$first"); done
  fi
  [[ ${#MPS_THREAD_PCT[@]} -eq $DP ]] || {
    echo "MPS_THREAD_PERCENTAGES must contain one value or exactly DP=$DP values" >&2
    exit 2
  }
fi
if [[ -n "$MPS_PINNED_MEM_LIMITS" ]]; then
  if [[ ${#MPS_PINNED_LIMIT[@]} -eq 1 && $DP -gt 1 ]]; then
    first=${MPS_PINNED_LIMIT[0]}
    MPS_PINNED_LIMIT=()
    for ((i = 0; i < DP; i++)); do MPS_PINNED_LIMIT+=("$first"); done
  fi
  [[ ${#MPS_PINNED_LIMIT[@]} -eq $DP ]] || {
    echo "MPS_PINNED_MEM_LIMITS must contain one value or exactly DP=$DP values" >&2
    exit 2
  }
fi
python3 - "$MEM_FRACTIONS" "$MPS_THREAD_PERCENTAGES" <<'PY'
import sys

for raw in sys.argv[1].split(","):
    if raw == "auto":
        continue
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"invalid MEM_FRACTIONS value: {raw!r}") from exc
    if not 0 < value < 1:
        raise SystemExit(f"MEM_FRACTIONS values must be in (0, 1), got {value}")
if sys.argv[2]:
    for raw in sys.argv[2].split(","):
        try:
            value = int(raw)
        except ValueError as exc:
            raise SystemExit(f"invalid MPS_THREAD_PERCENTAGES value: {raw!r}") from exc
        if not 1 <= value <= 100:
            raise SystemExit(
                f"MPS_THREAD_PERCENTAGES values must be in [1, 100], got {value}"
            )
PY

ONLINE_CPUS=""
if [[ -r /sys/devices/system/cpu/online ]]; then
  ONLINE_CPUS=$(</sys/devices/system/cpu/online)
fi
NUMA_CPUS=""
if [[ -r "/sys/devices/system/node/node${NUMA_NODE}/cpulist" ]]; then
  NUMA_CPUS=$(<"/sys/devices/system/node/node${NUMA_NODE}/cpulist")
elif [[ "$DRY_RUN" -eq 0 ]]; then
  echo "NUMA node $NUMA_NODE has no readable cpulist" >&2
  exit 2
fi
VALIDATE=(python3 "$HERE/summarize.py" validate-layout --dp "$DP"
  --server-core-sets "$SERVER_CORE_SETS" --client-core-sets "$CLIENT_CORE_SETS")
[[ -n "$ONLINE_CPUS" ]] && VALIDATE+=(--online-cpus "$ONLINE_CPUS")
[[ -n "$NUMA_CPUS" ]] && VALIDATE+=(--numa-cpus "$NUMA_CPUS")
[[ -n "$ROUTER_CORES" ]] && VALIDATE+=(--router-cores "$ROUTER_CORES")
"${VALIDATE[@]}" >/dev/null

if [[ "$DRY_RUN" -eq 0 ]]; then
  for command in nvidia-smi numactl curl setsid; do
    command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 1; }
  done
  if [[ -z "$GPU_UUID" ]]; then
    echo "GPU_UUID is required (use an nvidia-smi GPU-... UUID, not an ordinal)" >&2
    exit 2
  fi
  nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d ' ' | grep -Fxq "$GPU_UUID" || {
    echo "GPU_UUID $GPU_UUID is not visible to nvidia-smi" >&2
    exit 2
  }
  compute_mode=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader -i "$GPU_UUID" | head -n1 | tr -d ' ')
  [[ "$compute_mode" == Default ]] || {
    echo "GPU $GPU_UUID compute mode is '$compute_mode'; this matrix requires Default" >&2
    exit 2
  }
  if [[ "$REQUIRE_IDLE_GPU" -eq 1 ]]; then
    existing_pids=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null |
      awk -F, -v uuid="$GPU_UUID" '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if ($2 == uuid && $1 ~ /^[0-9]+$/) print $1}')
    [[ -z "$existing_pids" ]] || {
      echo "GPU $GPU_UUID already has compute clients: $existing_pids" >&2
      echo "use an idle GPU, or set REQUIRE_IDLE_GPU=0 only when interference is intentional" >&2
      exit 2
    }
  fi
  if [[ "$USE_MPS" -eq 1 ]]; then
    command -v nvidia-cuda-mps-control >/dev/null || {
      echo "nvidia-cuda-mps-control is required for USE_MPS=1" >&2
      exit 1
    }
  fi
fi

OUT="$OUT_ROOT/$LABEL"
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing condition directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT" "$OUT/workers"
COMMAND_LOG="$OUT/commands.sh"
: > "$COMMAND_LOG"

print_command() {
  printf '%q ' "$@" | tee -a "$COMMAND_LOG"
  printf '\n' | tee -a "$COMMAND_LOG"
}

SERVER_PGIDS=()
CLIENT_PIDS=()
ROUTER_PGID=""
MPS_OWNED=0
MPS_DIR_OWNED=0
MPS_CONTROL_PID=""
MPS_ROOT="$MPS_TMP_ROOT/$LABEL"
MPS_PIPE="$MPS_ROOT/pipe"
MPS_LOG="$MPS_ROOT/log"
NO_MPS_PIPE="$OUT/no-mps-pipe-does-not-exist"
if [[ "$USE_MPS" -eq 1 && ${#MPS_PIPE} -gt 80 ]]; then
  echo "MPS pipe path is too long (${#MPS_PIPE} chars): $MPS_PIPE" >&2
  echo "shorten MPS_TMP_ROOT or LABEL; CUDA MPS uses Unix-domain sockets" >&2
  exit 2
fi

mps_control() {
  env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control
}

group_pids() {
  local pgid=$1
  ps -eo pid=,pgid= | awk -v group="$pgid" '$2 == group {print $1}'
}

mps_ps() {
  printf 'ps\n' | mps_control
}

stop_group() {
  local pgid=$1
  [[ -z "$pgid" ]] && return
  kill -TERM -- "-$pgid" 2>/dev/null || true
}

pid_is_live() {
  local pid=$1 status
  kill -0 "$pid" 2>/dev/null || return 1
  status=$(ps -o stat= -p "$pid" 2>/dev/null || true)
  [[ -n "$status" && "$status" != Z* ]]
}

wait_group_exit() {
  local pgid=$1 timeout=$2
  local deadline=$((SECONDS + timeout))
  while kill -0 -- "-$pgid" 2>/dev/null; do
    ((SECONDS >= deadline)) && return 1
    sleep 1
  done
}

terminate_mps_clients_for_group() {
  local pgid=$1 server_pid client_pid
  [[ "$MPS_OWNED" -eq 1 ]] || return 0
  server_pid=$(printf 'get_server_list\n' | mps_control 2>/dev/null | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print; exit}')
  [[ -n "$server_pid" ]] || return 0
  local snapshot
  snapshot=$(mps_ps 2>/dev/null || true)
  while read -r client_pid; do
    [[ -n "$client_pid" ]] || continue
    if grep -Eq "(^|[^0-9])${client_pid}([^0-9]|$)" <<< "$snapshot"; then
      printf 'terminate_client %s %s\n' "$server_pid" "$client_pid" | mps_control >/dev/null 2>&1 || true
    fi
  done < <(group_pids "$pgid")
}

cleanup() {
  local status=$? client_stop_deadline pid
  trap - EXIT INT TERM
  for pid in "${CLIENT_PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  [[ -n "$ROUTER_PGID" ]] && stop_group "$ROUTER_PGID"
  for pgid in "${SERVER_PGIDS[@]}"; do stop_group "$pgid"; done

  client_stop_deadline=$((SECONDS + 10))
  for pid in "${CLIENT_PIDS[@]}"; do
    while pid_is_live "$pid" && ((SECONDS < client_stop_deadline)); do sleep 1; done
    pid_is_live "$pid" && kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  if [[ -n "$ROUTER_PGID" ]] && ! wait_group_exit "$ROUTER_PGID" 10; then
    kill -KILL -- "-$ROUTER_PGID" 2>/dev/null || true
  fi

  for pgid in "${SERVER_PGIDS[@]}"; do
    if ! wait_group_exit "$pgid" "$SHUTDOWN_TIMEOUT"; then
      echo "server group $pgid did not stop; asking owned MPS server to terminate its CUDA clients" >&2
      terminate_mps_clients_for_group "$pgid"
      sleep 2
      if ! wait_group_exit "$pgid" 10; then
        echo "server group $pgid still alive after terminate_client; forcing only this process group" >&2
        kill -KILL -- "-$pgid" 2>/dev/null || true
      fi
    fi
  done
  if [[ "$MPS_OWNED" -eq 1 ]]; then
    printf 'quit\n' | mps_control >/dev/null 2>&1 || true
    if [[ -n "$MPS_CONTROL_PID" ]]; then
      mps_stop_deadline=$((SECONDS + 10))
      while kill -0 "$MPS_CONTROL_PID" 2>/dev/null && \
        ((SECONDS < mps_stop_deadline)); do
        sleep 1
      done
    fi
    MPS_OWNED=0
  fi
  if [[ -d "$MPS_LOG" ]]; then
    mkdir -p "$OUT/mps_logs"
    cp -R "$MPS_LOG/." "$OUT/mps_logs/" 2>/dev/null || true
  fi
  if [[ "$MPS_DIR_OWNED" -eq 1 && -d "$MPS_ROOT" ]]; then
    if [[ -z "$MPS_CONTROL_PID" ]] || ! kill -0 "$MPS_CONTROL_PID" 2>/dev/null; then
      rm -rf "$MPS_ROOT"
      MPS_DIR_OWNED=0
    else
      echo "MPS control PID $MPS_CONTROL_PID is still alive; preserving $MPS_ROOT" >&2
    fi
  fi
  if [[ "$KEEP_AUDIO" -eq 0 && -d "$OUT" ]]; then
    find "$OUT" -type f -path '*/audio/*.wav' -delete
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "$USE_MPS" -eq 1 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
      -u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT CUDA_VISIBLE_DEVICES="$GPU_UUID" \
      CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
      nvidia-cuda-mps-control -d
  else
    if [[ -e "$MPS_ROOT" ]]; then
      echo "refusing to reuse existing MPS runtime directory: $MPS_ROOT" >&2
      exit 2
    fi
    mkdir -p "$MPS_PIPE" "$MPS_LOG"
    chmod 700 "$MPS_TMP_ROOT" "$MPS_ROOT" "$MPS_PIPE" "$MPS_LOG"
    MPS_DIR_OWNED=1
    mps_launch_status=0
    env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT \
      CUDA_VISIBLE_DEVICES="$GPU_UUID" CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
      CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d || \
      mps_launch_status=$?
    MPS_OWNED=1
    mps_ready=0
    mps_deadline=$((SECONDS + MPS_READY_TIMEOUT))
    while ((SECONDS < mps_deadline)); do
      if printf 'get_default_active_thread_percentage\n' | mps_control \
        > "$OUT/mps_control_check.txt" 2> "$OUT/mps_control_check.err" && \
        [[ -s "$OUT/mps_control_check.txt" ]] && \
        [[ -s "$MPS_PIPE/nvidia-cuda-mps-control.pid" ]]; then
        mps_ready=1
        break
      fi
      sleep 1
    done
    [[ "$mps_ready" -eq 1 ]] || {
      echo "MPS control daemon did not respond within ${MPS_READY_TIMEOUT}s (launch status $mps_launch_status)" >&2
      exit 1
    }
    MPS_CONTROL_PID=$(<"$MPS_PIPE/nvidia-cuda-mps-control.pid")
    [[ "$MPS_CONTROL_PID" =~ ^[1-9][0-9]*$ ]] || {
      echo "invalid MPS control PID: $MPS_CONTROL_PID" >&2
      exit 1
    }
  fi
fi

wait_health() {
  local port=$1 pgid=$2
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT))
  while ((SECONDS < deadline)); do
    if curl --silent --show-error --max-time 3 --fail "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    kill -0 -- "-$pgid" 2>/dev/null || return 1
    sleep 3
  done
  return 1
}

for ((i = 0; i < DP; i++)); do
  port=$((BASE_PORT + i))
  log="$OUT/workers/worker_${i}.server.log"
  env_args=(env -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY
    -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE -u CUDA_MPS_PINNED_DEVICE_MEM_LIMIT
    CUDA_VISIBLE_DEVICES="$GPU_UUID" PYTHONPATH="$REPO")
  if [[ "$USE_MPS" -eq 1 ]]; then
    env_args+=(CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG")
    [[ -n "$MPS_THREAD_PERCENTAGES" ]] && env_args+=(CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${MPS_THREAD_PCT[$i]}")
    [[ -n "$MPS_PINNED_MEM_LIMITS" ]] && env_args+=(CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="${MPS_PINNED_LIMIT[$i]}")
  else
    # NVIDIA documents a nonexistent pipe directory as the explicit MPS bypass.
    # This prevents an ambient/default MPS daemon from contaminating MPS=0.
    env_args+=(CUDA_MPS_PIPE_DIRECTORY="$NO_MPS_PIPE")
  fi
  command=(setsid "${env_args[@]}" numactl --physcpubind="${SERVER_CORES[$i]}"
    --membind="$NUMA_NODE" python -m sglang_omni.cli serve
    --model-path "$MODEL" --model-name "$MODEL_NAME" --host 127.0.0.1 --port "$port"
    --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
  [[ "${MEM_FRAC[$i]}" != auto ]] && command+=(--mem-fraction-static "${MEM_FRAC[$i]}")
  [[ -n "$MAX_TOTAL_TOKENS" ]] && command+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
  print_command "${command[@]}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    SERVER_PGIDS+=("dry-$i")
    continue
  fi
  "${command[@]}" > "$log" 2>&1 &
  pgid=$!
  SERVER_PGIDS+=("$pgid")
  echo "$pgid" > "$OUT/workers/worker_${i}.pgid"
  if ! wait_health "$port" "$pgid"; then
    echo "worker $i failed readiness on port $port; inspect $log" >&2
    exit 1
  fi
done

if [[ "$USE_MPS" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  mps_ps > "$OUT/mps_ps.txt"
  for ((i = 0; i < DP; i++)); do
    attached=0
    while read -r pid; do
      if grep -Eq "(^|[^0-9])${pid}([^0-9]|$)" "$OUT/mps_ps.txt"; then attached=1; break; fi
    done < <(group_pids "${SERVER_PGIDS[$i]}")
    [[ "$attached" -eq 1 ]] || {
      echo "worker $i has no process enumerated by MPS 'ps'; refusing an invalid comparison" >&2
      exit 1
    }
  done
fi

if [[ "$DRY_RUN" -eq 0 && ("$KV_EQUALITY" != off || "$CAPACITY_ONLY" -eq 1) ]]; then
  logs=()
  for ((i = 0; i < DP; i++)); do logs+=("$OUT/workers/worker_${i}.server.log"); done
  python3 "$HERE/summarize.py" extract-kv "${logs[@]}" > "$OUT/kv_capacity.json"
  kv_check=(python3 "$HERE/summarize.py" classify-kv "$OUT/kv_capacity.json")
  [[ -n "$MAX_TOTAL_TOKENS" ]] && kv_check+=(--expected "$MAX_TOTAL_TOKENS")
  kv_state=$("${kv_check[@]}")
  if [[ "$kv_state" != equal && "$kv_state" != exact ]]; then
    if [[ "$KV_EQUALITY" == require ]]; then
      echo "KV capacity gate failed before load: $kv_state; see $OUT/kv_capacity.json" >&2
      exit 1
    fi
    echo "warning: KV capacities are $kv_state; comparison is not memory-fair" >&2
  fi
fi

TARGET_PORTS=()
for ((i = 0; i < DP; i++)); do TARGET_PORTS+=("$((BASE_PORT + i))"); done
if [[ "$MODE" == router ]]; then
  worker_urls=()
  for port in "${TARGET_PORTS[@]}"; do worker_urls+=("http://127.0.0.1:$port"); done
  router_command=(setsid numactl --physcpubind="$ROUTER_CORES" --membind="$NUMA_NODE"
    python -m sglang_omni_router.serve --host 127.0.0.1 --port "$ROUTER_PORT"
    --worker-urls "${worker_urls[@]}" --model "$MODEL_NAME" --policy "$ROUTER_POLICY")
  print_command "${router_command[@]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "${router_command[@]}" > "$OUT/router.log" 2>&1 &
    ROUTER_PGID=$!
    wait_health "$ROUTER_PORT" "$ROUTER_PGID" || {
      echo "router failed readiness; inspect $OUT/router.log" >&2
      exit 1
    }
  fi
fi

write_manifest() {
  {
    echo "label=$LABEL"
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_sha=$(git -C "$REPO" rev-parse HEAD)"
    echo "git_status=$(git -C "$REPO" status --short | tr '\n' ';')"
    echo "model=$MODEL"
    echo "model_name=$MODEL_NAME"
    echo "meta=$META"
    echo "client_driver=$CLIENT_DRIVER"
    echo "tts_manifest=$TTS_MANIFEST"
    if [[ -n "$TTS_MANIFEST" ]]; then
      echo "tts_manifest_sha256=$(sha256sum "$TTS_MANIFEST" | awk '{print $1}')"
    fi
    echo "gpu_uuid=$GPU_UUID"
    echo "numa_node=$NUMA_NODE"
    echo "dp=$DP"
    echo "mps=$USE_MPS"
    echo "mode=$MODE"
    echo "server_core_sets=$SERVER_CORE_SETS"
    echo "client_core_sets=$CLIENT_CORE_SETS"
    echo "router_cores=$ROUTER_CORES"
    echo "router_policy=$ROUTER_POLICY"
    echo "mem_fractions=$MEM_FRACTIONS"
    echo "mps_thread_percentages=$MPS_THREAD_PERCENTAGES"
    echo "mps_pinned_mem_limits=$MPS_PINNED_MEM_LIMITS"
    echo "concurrency_per_worker=$CONCURRENCY_PER_WORKER"
    echo "max_running_requests=$MAX_RUNNING_REQUESTS"
    echo "cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS"
    echo "max_total_tokens=$MAX_TOTAL_TOKENS"
    echo "max_new_tokens=$MAX_NEW_TOKENS"
    echo "max_samples=$MAX_SAMPLES"
    echo "seed=$SEED"
    echo "warmup=$WARMUP"
    echo "lang=$BENCH_LANG"
    echo "ref_format=$REF_FORMAT"
    echo "allowed_local_media_path=$ALLOWED_LOCAL_MEDIA_PATH"
    echo "stream=$STREAM"
    echo "require_idle_gpu=$REQUIRE_IDLE_GPU"
    echo "capacity_only=$CAPACITY_ONLY"
    echo "keep_audio=$KEEP_AUDIO"
    echo "mps_tmp_root=$MPS_TMP_ROOT"
    echo "mps_runtime_root=$MPS_ROOT"
    echo "mps_control_pid=$MPS_CONTROL_PID"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      echo "uname=$(uname -a)"
      echo "python=$(python --version 2>&1)"
      echo "cuda_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU_UUID" | head -n1)"
      echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,compute_mode,pci.bus_id --format=csv,noheader -i "$GPU_UUID" | head -n1)"
      echo "topology_begin"
      nvidia-smi topo -m
      echo "topology_end"
    fi
  } > "$OUT/manifest.txt"
}
write_manifest

if [[ "$DRY_RUN" -eq 0 && "$CAPACITY_ONLY" -eq 1 ]]; then
  echo "capacity-only condition complete: $OUT"
  exit 0
fi

if [[ "$CLIENT_DRIVER" == seedtts ]]; then
  BENCH_COMMON=(python -m benchmarks.eval.benchmark_tts_seedtts
    --use-existing-server --generate-only --model "$MODEL_NAME" --meta "$META"
    --ref-format "$REF_FORMAT" --lang "$BENCH_LANG" --max-new-tokens "$MAX_NEW_TOKENS"
    --seed "$SEED" --warmup "$WARMUP" --disable-tqdm)
  [[ -n "$MAX_SAMPLES" ]] && BENCH_COMMON+=(--max-samples "$MAX_SAMPLES")
  [[ "$STREAM" -eq 1 ]] && BENCH_COMMON+=(--stream)
else
  BENCH_COMMON=(python -m benchmarks.same_gpu_dp.benchmark_tts_manifest
    --manifest "$TTS_MANIFEST" --model "$MODEL_NAME"
    --server-max-new-tokens "$MAX_NEW_TOKENS" --warmup "$WARMUP")
fi

CLIENT_BARRIER="$OUT/client_barrier"
RELEASE_FILE="$CLIENT_BARRIER/start"
[[ "$CLIENT_DRIVER" == seedtts ]] && mkdir -p "$CLIENT_BARRIER"

if [[ "$DRY_RUN" -eq 1 ]]; then
  for ((i = 0; i < DP; i++)); do
    if [[ "$MODE" == direct ]]; then
      target_port=${TARGET_PORTS[$i]}
      output="$OUT/workers/worker_${i}.benchmark"
    else
      target_port=$ROUTER_PORT
      output="$OUT/router_clients/client_${i}.benchmark"
    fi
    command=(numactl --physcpubind="${CLIENT_CORES[$i]}" \
      --membind="$NUMA_NODE" "${BENCH_COMMON[@]}" --port "$target_port" \
      --concurrency "$CONCURRENCY_PER_WORKER" --output-dir "$output")
    if [[ "$CLIENT_DRIVER" == seedtts ]]; then
      command+=(--sample-rotation-index "$i" --sample-rotation-count "$DP"
        --barrier-ready-file "$CLIENT_BARRIER/ready-$i"
        --barrier-release-file "$RELEASE_FILE")
    fi
    print_command "${command[@]}"
  done
  echo "dry run complete: commands and manifest are in $OUT"
  trap - EXIT INT TERM
  exit 0
fi

if [[ "$MODE" == direct ]]; then
  for ((i = 0; i < DP; i++)); do
    output="$OUT/workers/worker_${i}.benchmark"
    command=(numactl --physcpubind="${CLIENT_CORES[$i]}" --membind="$NUMA_NODE"
      "${BENCH_COMMON[@]}" --port "${TARGET_PORTS[$i]}"
      --concurrency "$CONCURRENCY_PER_WORKER" --output-dir "$output")
    if [[ "$CLIENT_DRIVER" == seedtts ]]; then
      command+=(--sample-rotation-index "$i" --sample-rotation-count "$DP"
        --barrier-ready-file "$CLIENT_BARRIER/ready-$i"
        --barrier-release-file "$RELEASE_FILE")
    fi
    print_command "${command[@]}"
    "${command[@]}" > "$OUT/workers/worker_${i}.benchmark.log" 2>&1 &
    CLIENT_PIDS+=("$!")
  done
else
  for ((i = 0; i < DP; i++)); do
    output="$OUT/router_clients/client_${i}.benchmark"
    mkdir -p "$OUT/router_clients"
    command=(numactl --physcpubind="${CLIENT_CORES[$i]}" --membind="$NUMA_NODE"
      "${BENCH_COMMON[@]}" --port "$ROUTER_PORT"
      --concurrency "$CONCURRENCY_PER_WORKER" --output-dir "$output")
    if [[ "$CLIENT_DRIVER" == seedtts ]]; then
      command+=(--sample-rotation-index "$i" --sample-rotation-count "$DP"
        --barrier-ready-file "$CLIENT_BARRIER/ready-$i"
        --barrier-release-file "$RELEASE_FILE")
    fi
    print_command "${command[@]}"
    "${command[@]}" > "$OUT/router_clients/client_${i}.benchmark.log" 2>&1 &
    CLIENT_PIDS+=("$!")
  done
fi

condition_wall_clock_s=""
if [[ "$CLIENT_DRIVER" == seedtts ]]; then
  deadline=$((SECONDS + CLIENT_READY_TIMEOUT))
  while :; do
    ready=0
    for ((i = 0; i < DP; i++)); do
      [[ -e "$CLIENT_BARRIER/ready-$i" ]] && ready=$((ready + 1))
    done
    [[ "$ready" -eq "$DP" ]] && break
    for ((i = 0; i < DP; i++)); do
      pid=${CLIENT_PIDS[$i]}
      if ! pid_is_live "$pid"; then
        echo "benchmark client $i exited before the start barrier; inspect its benchmark log" >&2
        exit 1
      fi
    done
    ((SECONDS < deadline)) \
      || { echo "benchmark clients did not reach the start barrier" >&2; exit 1; }
    sleep 1
  done
  condition_start_ns=$(python3 -c 'import time; print(time.monotonic_ns())')
  : > "$RELEASE_FILE"
fi
client_failed=0
for pid in "${CLIENT_PIDS[@]}"; do wait "$pid" || client_failed=1; done
if [[ "$CLIENT_DRIVER" == seedtts ]]; then
  condition_end_ns=$(python3 -c 'import time; print(time.monotonic_ns())')
  condition_wall_clock_s=$(python3 - "$condition_start_ns" "$condition_end_ns" <<'PY'
import sys

print((int(sys.argv[2]) - int(sys.argv[1])) / 1_000_000_000)
PY
  )
  printf '%s\n' "$condition_wall_clock_s" > "$OUT/condition_wall_clock_s.txt"
fi
[[ "$client_failed" -eq 0 ]] || { echo "one or more benchmark clients failed" >&2; exit 1; }

if [[ "$MODE" == router ]]; then
  curl --silent --show-error --fail "http://127.0.0.1:${ROUTER_PORT}/workers" \
    > "$OUT/router_workers_after.json"
  python3 "$HERE/summarize.py" summarize-router \
    --output "$OUT/router_summary.json" "$OUT/router_workers_after.json"
fi

if [[ "$MODE" == direct ]]; then
  result_paths=()
  logs=()
  for ((i = 0; i < DP; i++)); do
    result_paths+=("$OUT/workers/worker_${i}.benchmark/speed_results.json")
    logs+=("$OUT/workers/worker_${i}.server.log")
  done
else
  result_paths=()
  for ((i = 0; i < DP; i++)); do
    result_paths+=("$OUT/router_clients/client_${i}.benchmark/speed_results.json")
  done
fi
summary_command=(python3 "$HERE/summarize.py" summarize --output "$OUT/summary.json")
[[ -n "$condition_wall_clock_s" ]] \
  && summary_command+=(--wall-clock-s "$condition_wall_clock_s")
summary_command+=("${result_paths[@]}")
"${summary_command[@]}"

echo "condition complete: $OUT"
