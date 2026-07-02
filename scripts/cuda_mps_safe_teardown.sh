#!/usr/bin/env bash
set -euo pipefail

# Safely stop a per-run CUDA MPS daemon after benchmark/server teardown.
#
# This script is intentionally strict about the pipe directory. It is meant for
# benchmark recipes that start MPS with a unique CUDA_MPS_PIPE_DIRECTORY. Refuse
# the default shared pipe unless ALLOW_DEFAULT_MPS_PIPE=1 is set.

MPS_PIPE="${MPS_PIPE:-${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}}"
MPS_LOG="${MPS_LOG:-${CUDA_MPS_LOG_DIRECTORY:-}}"
ALLOW_DEFAULT_MPS_PIPE="${ALLOW_DEFAULT_MPS_PIPE:-0}"
CLIENT_GRACE_SECS="${CLIENT_GRACE_SECS:-5}"

if [[ "${MPS_PIPE}" == "/tmp/nvidia-mps" && "${ALLOW_DEFAULT_MPS_PIPE}" != "1" ]]; then
  echo "Refusing to stop shared default MPS pipe ${MPS_PIPE}." >&2
  echo "Set CUDA_MPS_PIPE_DIRECTORY to a unique per-run directory, or set ALLOW_DEFAULT_MPS_PIPE=1." >&2
  exit 2
fi

if [[ ! -S "${MPS_PIPE}/control" && ! -p "${MPS_PIPE}/control" ]]; then
  echo "No MPS control socket/pipe found under ${MPS_PIPE}; nothing to stop."
  exit 0
fi

export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE}"
if [[ -n "${MPS_LOG}" ]]; then
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG}"
fi

mps_control() {
  local command="$1"
  printf '%s\n' "${command}" | nvidia-cuda-mps-control 2>/dev/null || true
}

mapfile -t clients < <(
  mps_control "ps" |
    awk '$1 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {print $1 " " $3}' |
    sort -u
)

for entry in "${clients[@]}"; do
  client_pid="${entry%% *}"
  server_pid="${entry##* }"
  echo "Terminating MPS client context client_pid=${client_pid} server_pid=${server_pid}"
  result="$(mps_control "terminate_client ${server_pid} ${client_pid}" | tail -n 1 | tr -d '[:space:]')"
  if [[ "${result}" != "0" && "${result}" != "CUDA_SUCCESS" ]]; then
    echo "Warning: terminate_client returned ${result:-<empty>} for client_pid=${client_pid}" >&2
  fi
done

if [[ "${#clients[@]}" -gt 0 ]]; then
  sleep "${CLIENT_GRACE_SECS}"
fi

for entry in "${clients[@]}"; do
  client_pid="${entry%% *}"
  if kill -0 "${client_pid}" 2>/dev/null; then
    echo "Killing terminated MPS client process pid=${client_pid}"
    kill -TERM "${client_pid}" 2>/dev/null || true
  fi
done

sleep 2

for entry in "${clients[@]}"; do
  client_pid="${entry%% *}"
  if kill -0 "${client_pid}" 2>/dev/null; then
    echo "Force killing terminated MPS client process pid=${client_pid}"
    kill -KILL "${client_pid}" 2>/dev/null || true
  fi
done

echo "Stopping MPS control daemon for pipe ${MPS_PIPE}"
mps_control "quit" >/dev/null
sleep 2

if [[ -n "${MPS_LOG}" ]]; then
  rm -rf "${MPS_PIPE}" "${MPS_LOG}"
else
  rm -rf "${MPS_PIPE}"
fi

echo "MPS teardown complete."
