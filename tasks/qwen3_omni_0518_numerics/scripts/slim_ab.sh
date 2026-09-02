#!/usr/bin/env bash
# The slim A/B: three workloads on manually started servers, no router, no
# pytest. Per arm: the voice clone bench at c1, c16 and c32 on the bf16
# colocated profile (one GPU), MMSU at c16 on the bf16 thinker profile (one
# GPU), Video-MME with the talker at c16 on the bf16 disagg topology (two
# GPUs). Then `score` runs the CI's WER and UTMOS scorers on the voice
# clone outputs against a Qwen3-ASR server. Results land in
# $OUT/<stage>/<arm>/ so full_ab_compare.py reads them.
#
# Required: OMNI_ROOT (clean tree, installed editable), OUT (outside the
# tree), GPU (two GPUs, default 0,1). Optional: A_SHA, B_SHA.
# Run from a copy outside the tree:
#   cp -r "$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts" "$OUT/scripts"
#   OMNI_ROOT=... OUT=... GPU=0,1 bash "$OUT/scripts/slim_ab.sh"
#   OMNI_ROOT=... OUT=... GPU=0   bash "$OUT/scripts/slim_ab.sh" score
#   python "$OUT/scripts/full_ab_compare.py" "$OUT" --md "$OUT/readout.md"
set -uo pipefail

: "${OMNI_ROOT:?set OMNI_ROOT}"
: "${OUT:?set OUT}"
GPU="${GPU:-0,1}"
A_SHA="${A_SHA:-68c88dae6}"
B_SHA="${B_SHA:-9769867a0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
# shellcheck source=/dev/null
source "$SCRIPT_DIR/h100_runs.sh"
RUN_BENCH="$SCRIPT_DIR/run_bench.py"
SCRIPTS="$SCRIPT_DIR"
TOP_LOGPROBS=0
GPU_ONE=${GPU%%,*}

log() { echo "$(date -Is) $*" | tee -a "$OUT/slim_ab.log"; }

gpus_idle() {
  OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB=2048 \
  OMNI_CI_GPU_CLEAN_WAIT_SECONDS=600 \
  OMNI_CI_GPU_CLEAN_POLL_SECONDS=5 \
  bash "$OMNI_ROOT/.github/scripts/delete_gpu_process.sh" --kill-orphans >> "$OUT/slim_ab.log" 2>&1 || return 1
  sleep 3
}

checkout_arm() {
  local arm=$1 sha
  sha=$([ "$arm" = A ] && echo "$A_SHA" || echo "$B_SHA")
  git -C "$OMNI_ROOT" checkout -q --detach "$sha" || return 1
  git -C "$OMNI_ROOT" rev-parse HEAD
}

bench() {
  # $1 stage dir name, $2 arm, $3 run_bench task, $4 port, $5 concurrency
  local dir="$OUT/$1/$2"
  mkdir -p "$dir"
  echo "A B" > "$OUT/$1/order.txt"
  git -C "$OMNI_ROOT" rev-parse HEAD > "$dir/git_head.txt"
  (cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" "$3" \
    --port "$4" --out "$dir" --concurrency "$5" --top-logprobs 0) > "$dir/bench.log" 2>&1
  echo $? > "$dir/exit_code"
  log "$1 $2 exit $(cat "$dir/exit_code")"
}

serve_logs() {
  # copy the serve log of the boot into the stage dir for the readout
  cp "$OUT/logs/$1" "$OUT/$2/$3/$1" 2>/dev/null || true
}

run_arm() {
  local arm=$1
  checkout_arm "$arm" > /dev/null || return 1
  log "arm $arm at $(git -C "$OMNI_ROOT" rev-parse --short HEAD)"

  gpus_idle || return 1
  GPU=$GPU_ONE serve_bf16_colocated 31000 || return 1
  for c in 1 16 32; do bench "seedtts_c$c" "$arm" seedtts-vc 31000 "$c"; done
  stop_server 31000
  for c in 1 16 32; do serve_logs serve_bf16_colocated_31000.log "seedtts_c$c" "$arm"; done

  gpus_idle || return 1
  GPU=$GPU_ONE serve_bf16_thinker 31001 || return 1
  bench mmsu_c16 "$arm" mmsu 31001 16
  stop_server 31001
  serve_logs serve_bf16_thinker_31001.log mmsu_c16 "$arm"

  gpus_idle || return 1
  GPU=$GPU serve_bf16_disagg 31002 || return 1
  bench videomme_talker_c16 "$arm" videomme-talker 31002 16
  stop_server 31002
  serve_logs serve_bf16_disagg_31002.log videomme_talker_c16 "$arm"
}

run_score() {
  local port=31011 dir
  gpus_idle || return 1
  GPU=$GPU_ONE _launch_server $port "serve_asr_$port.log" sgl-omni serve \
    --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $port || return 1
  for dir in "$OUT"/seedtts_c*/[AB]/seedtts_vc; do
    [ -f "$dir/speed_results.json" ] || continue
    log "score $dir"
    (cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python -m benchmarks.eval.benchmark_omni_seedtts \
      --transcribe-only --meta zhaochenyang20/seed-tts-eval-50-arrow --output-dir "$dir" \
      --model qwen3-omni --lang en --port $port) > "$dir/../transcribe.log" 2>&1
    (cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python -m benchmarks.eval.benchmark_omni_seedtts \
      --utmos-only --output-dir "$dir" --device "cuda:0") > "$dir/../utmos.log" 2>&1
  done
  stop_server $port
}

main() {
  mkdir -p "$OUT/logs"
  case "$SCRIPT_DIR/" in
    "$(cd "$OMNI_ROOT" && pwd)"/*) echo "run from a copy outside OMNI_ROOT" >&2; exit 1 ;;
  esac
  if [ -n "$(git -C "$OMNI_ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "OMNI_ROOT has uncommitted changes" >&2
    exit 1
  fi
  local orig_ref
  orig_ref=$(git -C "$OMNI_ROOT" symbolic-ref -q --short HEAD || git -C "$OMNI_ROOT" rev-parse HEAD)
  trap 'git -C "$OMNI_ROOT" checkout -q "$orig_ref"; log "restored $orig_ref"' EXIT
  capture_env
  if [ "${1:-}" = score ]; then
    run_score
    return
  fi
  run_arm A
  run_arm B
  log "done, next: bash $SCRIPT_DIR/slim_ab.sh score, then python $SCRIPT_DIR/full_ab_compare.py $OUT"
}

main "$@"
