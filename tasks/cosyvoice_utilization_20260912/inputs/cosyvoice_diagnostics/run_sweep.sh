#!/usr/bin/env bash
# Run from the SGLang-Omni checkout, in the benchmark environment.
# Does NOT launch/stop/modify a server. Use an approved test instance.
set -euo pipefail
LABEL="${1:?Usage: run_sweep.sh LABEL [streaming|buffered]}"
MODE="${2:-streaming}"
case "$MODE" in streaming|buffered) ;; *) echo 'Mode must be streaming or buffered' >&2; exit 2;; esac
if [[ ! "$LABEL" =~ ^[a-zA-Z0-9_.-]+$ ]]; then echo 'LABEL: use letters, digits, dot, underscore, hyphen' >&2; exit 2; fi
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
PORT="${PORT:-8000}"
CONCURRENCIES="${CONCURRENCIES:-1 2 4 8 16}"
REPEATS="${REPEATS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
WARMUP="${WARMUP:-32}"
OUT="${OUT:-results/cosyvoice_diagnostics}"
for value in "$PORT" "$REPEATS" "$MAX_SAMPLES"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then echo 'PORT, REPEATS, and MAX_SAMPLES must be positive integers' >&2; exit 2; fi
done
if [[ ! "$WARMUP" =~ ^[0-9]+$ ]]; then echo 'WARMUP must be a nonnegative integer' >&2; exit 2; fi
ARGS=(--model "$MODEL" --port "$PORT" --lang en --use-existing-server
      --generate-only --max-samples "$MAX_SAMPLES" --warmup "$WARMUP")
[[ "$MODE" == streaming ]] && ARGS+=(--stream)
for ((run=1; run<=REPEATS; run++)); do
  for concurrency in $CONCURRENCIES; do
    if [[ ! "$concurrency" =~ ^[1-9][0-9]*$ ]]; then echo 'Invalid concurrency' >&2; exit 2; fi
    target="$OUT/$LABEL/$MODE/run_$run/c$concurrency"
    if [[ -e "$target" ]]; then echo "Refusing to reuse result directory: $target" >&2; exit 2; fi
    mkdir -p "$target"
    "$PYTHON" -m benchmarks.eval.benchmark_tts_seedtts "${ARGS[@]}" \
      --max-concurrency "$concurrency" --output-dir "$target" \
      2>&1 | tee "$target/console.log"
  done
done
