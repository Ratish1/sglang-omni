#!/usr/bin/env bash
set -euo pipefail

# Probe Higgs TTS throughput across direct worker, manual DP split, router DP,
# and >max_running_requests admission. Start workers/routers before running.

MODEL="${MODEL:-bosonai/higgs-audio-v3-tts-4b}"
META="${META:-zhaochenyang20/seed-tts-eval-arrow}"
BENCH_LANG="${BENCH_LANG:-${HIGGS_TTS_BENCH_LANG:-en}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/higgs_tts_router_mps_probe/$(date +%Y%m%d_%H%M%S)}"

MODE="${MODE:-all}" # all | direct | admission | manual | router | multi-router
WORKER_URLS="${WORKER_URLS:-http://127.0.0.1:8011}"
ROUTER_URL="${ROUTER_URL-http://127.0.0.1:8008}"
MULTI_ROUTER_URLS="${MULTI_ROUTER_URLS:-}"

WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-64}"
ADMISSION_CONCURRENCIES="${ADMISSION_CONCURRENCIES:-65}"
WARMUP="${WARMUP:-1}"

POLL_INTERVAL_S="${POLL_INTERVAL_S:-2}"
ROUTER_PID="${ROUTER_PID:-}"

read -r -a WORKERS <<<"${WORKER_URLS}"
read -r -a ROUTERS <<<"${MULTI_ROUTER_URLS}"

mkdir -p "${RESULTS_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

bench() {
  local label="$1"
  local url="$2"
  local concurrency="$3"
  local out="${RESULTS_ROOT}/${label}"
  mkdir -p "${out}"
  log "benchmark ${label}: url=${url} concurrency=${concurrency}"
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --generate-only \
    --use-existing-server \
    --base-url "${url}" \
    --model "${MODEL}" \
    --meta "${META}" \
    --ref-format references \
    --output-dir "${out}" \
    --lang "${BENCH_LANG}" \
    --concurrency "${concurrency}" \
    --warmup "${WARMUP}" \
    --disable-tqdm \
    2>&1 | tee "${out}/benchmark.log"
}

poll_url() {
  local label="$1"
  local url="$2"
  local endpoint="$3"
  local out="$4"
  mkdir -p "$(dirname "${out}")"
  while true; do
    {
      printf '\n--- %s %s %s ---\n' "$(date -Is)" "${label}" "${endpoint}"
      if [[ -n "${SGLANG_OMNI_ADMIN_KEY:-}" ]]; then
        curl -sS --max-time 5 \
          -H "Authorization: Bearer ${SGLANG_OMNI_ADMIN_KEY}" \
          "${url}${endpoint}" || true
      else
        curl -sS --max-time 5 "${url}${endpoint}" || true
      fi
      printf '\n'
    } >>"${out}"
    sleep "${POLL_INTERVAL_S}"
  done
}

router_pid_tree() {
  local roots=("$@")
  local all=()
  local queue=()
  local pid child

  for pid in "${roots[@]}"; do
    [[ -n "${pid}" ]] || continue
    all+=("${pid}")
    queue+=("${pid}")
  done

  while [[ "${#queue[@]}" -gt 0 ]]; do
    pid="${queue[0]}"
    queue=("${queue[@]:1}")
    while read -r child; do
      [[ -n "${child}" ]] || continue
      all+=("${child}")
      queue+=("${child}")
    done < <(pgrep -P "${pid}" 2>/dev/null || true)
  done

  local IFS=,
  printf '%s' "${all[*]}"
}

start_common_monitors() {
  local label="$1"
  mkdir -p "${RESULTS_ROOT}/${label}/monitor"
  nvidia-smi dmon -s pucm -d 1 -o DT >"${RESULTS_ROOT}/${label}/monitor/nvidia_smi_dmon.log" 2>&1 &
  MONITOR_PIDS+=("$!")

  for idx in "${!WORKERS[@]}"; do
    poll_url "worker-${idx}" "${WORKERS[$idx]}" "/model_info" \
      "${RESULTS_ROOT}/${label}/monitor/worker_${idx}_model_info.jsonl" &
    MONITOR_PIDS+=("$!")
  done

  if [[ -n "${ROUTER_URL}" && "${label}" == "router" ]]; then
    poll_url "router" "${ROUTER_URL}" "/workers" \
      "${RESULTS_ROOT}/${label}/monitor/router_workers.jsonl" &
    MONITOR_PIDS+=("$!")
  fi

  if [[ -n "${ROUTER_PID}" ]] && command -v pidstat >/dev/null 2>&1; then
    pidstat -t -p "$(router_pid_tree "${ROUTER_PID}")" 1 >"${RESULTS_ROOT}/${label}/monitor/router_pidstat.log" 2>&1 &
    MONITOR_PIDS+=("$!")
  fi
}

stop_monitors() {
  for pid in "${MONITOR_PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  wait "${MONITOR_PIDS[@]:-}" >/dev/null 2>&1 || true
  MONITOR_PIDS=()
}

run_with_monitors() {
  local label="$1"
  shift
  MONITOR_PIDS=()
  start_common_monitors "${label}"
  set +e
  "$@"
  local status=$?
  set -e
  stop_monitors
  return "${status}"
}

case_direct() {
  bench "direct_c${WORKER_CONCURRENCY}" "${WORKERS[0]}" "${WORKER_CONCURRENCY}"
}

case_admission() {
  for c in ${ADMISSION_CONCURRENCIES}; do
    bench "admission_c${c}" "${WORKERS[0]}" "${c}"
  done
}

case_manual() {
  local pids=()
  for idx in "${!WORKERS[@]}"; do
    bench "manual_worker${idx}_c${WORKER_CONCURRENCY}" \
      "${WORKERS[$idx]}" "${WORKER_CONCURRENCY}" &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
  done
  return "${status}"
}

case_router() {
  local nworkers="${#WORKERS[@]}"
  local router_concurrency="${ROUTER_CONCURRENCY:-$((WORKER_CONCURRENCY * nworkers))}"
  bench "router_c${router_concurrency}" "${ROUTER_URL}" "${router_concurrency}"
}

case_multi_router() {
  if [[ "${#ROUTERS[@]}" -eq 0 ]]; then
    log "MULTI_ROUTER_URLS is empty; skipping multi-router case"
    return 0
  fi
  local pids=()
  for idx in "${!ROUTERS[@]}"; do
    bench "multi_router${idx}_c${WORKER_CONCURRENCY}" \
      "${ROUTERS[$idx]}" "${WORKER_CONCURRENCY}" &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
  done
  return "${status}"
}

log "results: ${RESULTS_ROOT}"
log "mode=${MODE} model=${MODEL} lang=${BENCH_LANG} workers=${WORKER_URLS} router=${ROUTER_URL}"

CASE_STATUSES=()

run_case() {
  local label="$1"
  shift
  set +e
  run_with_monitors "${label}" "$@"
  local status=$?
  set -e
  CASE_STATUSES+=("${label}:${status}")
  printf '%s\t%s\n' "${label}" "${status}" >>"${RESULTS_ROOT}/case_status.tsv"
  if [[ "${status}" -ne 0 ]]; then
    log "case ${label} exited with status ${status}; continuing"
  fi
  return 0
}

case "${MODE}" in
  all)
    run_case direct case_direct
    run_case admission case_admission
    run_case manual case_manual
    run_case router case_router
    run_case multi-router case_multi_router
    ;;
  direct)
    run_case direct case_direct
    ;;
  admission)
    run_case admission case_admission
    ;;
  manual)
    run_case manual case_manual
    ;;
  router)
    run_case router case_router
    ;;
  multi-router)
    run_case multi-router case_multi_router
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

log "case statuses: ${CASE_STATUSES[*]}"
log "done: ${RESULTS_ROOT}"
