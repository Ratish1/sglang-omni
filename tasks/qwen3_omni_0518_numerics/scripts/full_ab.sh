#!/usr/bin/env bash
# One A/B over every Qwen3-Omni CI stage, both arms in one container.
#
# Each stage is the CI test file run with pytest at the CI settings, so the
# thresholds the tests assert are the verdict of that arm, and the result
# JSONs, the server logs and the pytest log of every run stay under
# $OUT/<stage>/<arm>/. The two arms alternate order from stage to stage
# (A then B, B then A, ...), GPUs are cleaned with the CI script between
# runs, and the checkout is switched with git checkout --detach per arm and
# restored at the end. full_ab_compare.py prints the side by side readout.
#
# Required environment:
#   OMNI_ROOT   the checkout mounted in the container, clean working tree,
#               installed editable (the server imports the installed package)
#   OUT         result directory, outside OMNI_ROOT
#   GPU         CUDA_VISIBLE_DEVICES value, two GPUs (default 0,1)
# Optional:
#   A_SHA, B_SHA   the arms (default 68c88dae6 and ce2735c02)
#   STAGES         space separated subset of the stage names below
#   SGLANG_SEEDTTS50_DIR, SEEDTTS_SIM_CACHE_DIR   as the CI tts stage sets them
#
# Run from a copy outside the tree, the checkout changes under a script that
# lives in it:
#   cp -r "$OMNI_ROOT/tasks/qwen3_omni_0518_numerics/scripts" "$OUT/scripts"
#   OMNI_ROOT=... OUT=... GPU=0,1 bash "$OUT/scripts/full_ab.sh"
#   python "$OUT/scripts/full_ab_compare.py" "$OUT" --md "$OUT/readout.md"
#
# Then the c1 addendum for the tts stage (two manual boots, section 5 of
# 09_reservation_flow_full_ab.md):
#   OMNI_ROOT=... OUT=... GPU=0 bash "$OUT/scripts/full_ab.sh" tts_c1
set -uo pipefail

: "${OMNI_ROOT:?set OMNI_ROOT}"
: "${OUT:?set OUT}"
GPU="${GPU:-0,1}"
A_SHA="${A_SHA:-68c88dae6}"
B_SHA="${B_SHA:-ce2735c02}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CI order (.github/workflows/test-qwen3-omni-ci.yaml).
ALL_STAGES=(
  "thinker_length tests/test_model/test_qwen3_omni_thinker_length.py"
  "tts tests/test_model/test_qwen3_omni_tts_ci.py"
  "mmmu tests/test_model/test_qwen3_omni_mmmu_ci.py"
  "mmmu_talker tests/test_model/test_qwen3_omni_mmmu_talker_ci.py"
  "mmsu tests/test_model/test_qwen3_omni_mmsu_ci.py"
  "mmsu_talker tests/test_model/test_qwen3_omni_mmsu_talker_ci.py"
  "videomme tests/test_model/test_qwen3_omni_videomme_ci.py"
  "videomme_talker tests/test_model/test_qwen3_omni_videomme_talker_ci.py"
  "videoamme tests/test_model/test_qwen3_omni_videoamme_ci.py"
  "videoamme_talker_tp2 tests/test_model/test_qwen3_omni_videoamme_talker_tp2_ci.py"
)

# Other models whose stages run the same scheduler, only when named in
# STAGES (STAGES="moss_td qwen3_asr qwen3_tts"). One GPU each, the fixtures
# pick device 0 of the visible set.
EXTRA_STAGES=(
  "moss_td tests/test_model/test_asr_ci_multi_speaker.py"
  "qwen3_asr tests/test_model/test_asr_ci_seedtts.py --asr-ci-model qwen3"
  "qwen3_tts tests/test_model/test_tts_ci.py --tts-ci-model qwen3-tts"
)

# Workflow level env of the CI (test-qwen3-omni-ci.yaml).
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0

log() { echo "$(date -Is) $*" | tee -a "$OUT/full_ab.log"; }

guard() {
  case "$SCRIPT_DIR/" in
    "$(cd "$OMNI_ROOT" && pwd)"/*)
      echo "run this script from a copy outside OMNI_ROOT, see the header" >&2
      return 1 ;;
  esac
  if [ -n "$(git -C "$OMNI_ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "OMNI_ROOT has uncommitted changes, commit or discard them first" >&2
    return 1
  fi
  local location
  location=$(cd / && python -c 'import os, sglang_omni; print(os.path.dirname(os.path.dirname(os.path.abspath(sglang_omni.__file__))))')
  if [ "$location" != "$(cd "$OMNI_ROOT" && pwd)" ]; then
    echo "the server would import $location, not OMNI_ROOT; run: python -m pip install -e \"$OMNI_ROOT\" --no-deps" >&2
    return 1
  fi
  for sha in "$A_SHA" "$B_SHA"; do
    git -C "$OMNI_ROOT" cat-file -e "$sha^{commit}" 2>/dev/null || {
      echo "commit $sha is not in OMNI_ROOT, fetch it first" >&2
      return 1
    }
  done
}

capture_env() {
  mkdir -p "$OUT/env"
  python -m pip freeze > "$OUT/env/pip_freeze.txt" 2>/dev/null
  python - > "$OUT/env/torch.txt" 2>&1 <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
for name in ("sglang", "sgl_kernel", "flashinfer", "transformers"):
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "?"))
    except Exception as exc:
        print(name, "import failed:", exc)
PY
  nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv > "$OUT/env/nvidia_smi.txt" 2>&1
  env | grep -E '^(SGLANG_|SGL_|TRITON_|FLASHINFER_|TORCH|CUDA_|NCCL_|OMNI_|HF_|SEEDTTS_|PYTORCH_)' | sort > "$OUT/env/env.txt"
  echo "A $A_SHA $(git -C "$OMNI_ROOT" log -1 --format=%s "$A_SHA")" > "$OUT/env/arms.txt"
  echo "B $B_SHA $(git -C "$OMNI_ROOT" log -1 --format=%s "$B_SHA")" >> "$OUT/env/arms.txt"
}

gpus_idle() {
  OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB=2048 \
  OMNI_CI_GPU_CLEAN_WAIT_SECONDS=600 \
  OMNI_CI_GPU_CLEAN_POLL_SECONDS=5 \
  bash "$OMNI_ROOT/.github/scripts/delete_gpu_process.sh" --kill-orphans >> "$OUT/full_ab.log" 2>&1 || return 1
  sleep 3
}

checkout_arm() {
  local arm=$1 sha
  sha=$([ "$arm" = A ] && echo "$A_SHA" || echo "$B_SHA")
  git -C "$OMNI_ROOT" checkout -q --detach "$sha" || return 1
  git -C "$OMNI_ROOT" rev-parse HEAD
}

run_stage_arm() {
  # $2 is the test file followed by its pytest options, split on spaces.
  local name=$1 testargs=$2 arm=$3
  local dir="$OUT/$name/$arm"
  mkdir -p "$dir/tmp"
  checkout_arm "$arm" > "$dir/git_head.txt" || return 1
  gpus_idle || { echo "gpu not idle" > "$dir/exit_code"; return 1; }
  log "start $name $arm $(cat "$dir/git_head.txt")"
  # GITHUB_ACTIONS=true makes the fixtures write each server's log to a
  # server.log under the basetemp (benchmarks/benchmarker/utils.py:67-72)
  # instead of streaming it into the pytest output.
  (
    cd "$OMNI_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$OMNI_ROOT" GITHUB_ACTIONS=true \
      python -m pytest $testargs -v -s -x -p no:cacheprovider --basetemp "$dir/tmp" \
      > "$dir/pytest.log" 2>&1
  )
  local code=$?
  echo "$code" > "$dir/exit_code"
  nvidia-smi --query-gpu=index,memory.used --format=csv > "$dir/nvidia_smi_after.txt" 2>&1
  log "end $name $arm exit $code"
  gpus_idle || true
  return "$code"
}

run_stage() {
  local name=$1 testargs=$2 index=$3 order
  if [ $((index % 2)) -eq 0 ]; then order="A B"; else order="B A"; fi
  mkdir -p "$OUT/$name"
  echo "$order" > "$OUT/$name/order.txt"
  if [ "$name" = tts ]; then
    (cd "$OMNI_ROOT" && python -m benchmarks.metrics.speaker_similarity_assets --warm-cache) >> "$OUT/full_ab.log" 2>&1
  fi
  for arm in $order; do
    run_stage_arm "$name" "$testargs" "$arm" || true
  done
}

# c1 addendum: the voice clone bench at concurrency 1 on the bf16 colocated
# profile, one manual boot per arm, through h100_runs.sh from this copy.
run_tts_c1() {
  local port=31010
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/h100_runs.sh"
  RUN_BENCH="$SCRIPT_DIR/run_bench.py"
  SCRIPTS="$SCRIPT_DIR"
  mkdir -p "$OUT/tts_c1"
  echo "A B" > "$OUT/tts_c1/order.txt"
  for arm in A B; do
    local dir="$OUT/tts_c1/$arm"
    mkdir -p "$dir"
    checkout_arm "$arm" > "$dir/git_head.txt" || return 1
    gpus_idle || return 1
    log "start tts_c1 $arm $(cat "$dir/git_head.txt")"
    OUT="$dir" serve_bf16_colocated $port || { echo 1 > "$dir/exit_code"; continue; }
    (cd "$OMNI_ROOT" && PYTHONPATH="$OMNI_ROOT" python "$RUN_BENCH" seedtts-vc \
      --port $port --out "$dir" --concurrency 1) > "$dir/bench.log" 2>&1
    echo $? > "$dir/exit_code"
    OUT="$dir" stop_server $port
    log "end tts_c1 $arm exit $(cat "$dir/exit_code")"
  done
}

main() {
  mkdir -p "$OUT"
  guard || exit 1
  local orig_ref
  orig_ref=$(git -C "$OMNI_ROOT" symbolic-ref -q --short HEAD || git -C "$OMNI_ROOT" rev-parse HEAD)
  trap 'git -C "$OMNI_ROOT" checkout -q "$orig_ref"; log "restored $orig_ref"' EXIT
  capture_env
  if [ "${1:-}" = tts_c1 ]; then
    run_tts_c1
    return
  fi
  local wanted="${STAGES:-}" index=0 name testargs
  for entry in "${ALL_STAGES[@]}" "${EXTRA_STAGES[@]}"; do
    name=${entry%% *}
    testargs=${entry#* }
    if [ -n "$wanted" ]; then
      [[ " $wanted " == *" $name "* ]] || continue
    elif [[ " ${EXTRA_STAGES[*]} " == *" $entry "* ]]; then
      continue
    fi
    run_stage "$name" "$testargs" "$index"
    index=$((index + 1))
  done
  log "done, readout: python $SCRIPT_DIR/full_ab_compare.py $OUT --md $OUT/readout.md"
}

main "$@"
