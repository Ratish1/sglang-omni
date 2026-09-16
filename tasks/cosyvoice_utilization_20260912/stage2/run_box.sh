#!/usr/bin/env bash
# One command for a stage 2 box experiment: sync both worktrees, put the tree
# under test on the path, record provenance, run, archive.
#
#   ./run_box.sh g0_hop_cache_numerics.py
#
# The environment the 2026-09-16 run used, overridable:
#   MAIN_REV     commit of sglang-omni under test           cc85ddaa9
#   MAIN_WT      worktree for it                            /sgl-workspace/wt/cosy-main
#   ANALYSIS_WT  worktree for these scripts                 /sgl-workspace/wt/cosyvoice-analysis
#   COSYVOICE    CosyVoice clone with its Matcha submodule  /sgl-workspace/CosyVoice-utilization-20260912
#   REPO         checkout the worktrees hang off            /sgl-workspace/sglang-omni
set -euo pipefail

SCRIPT=${1:?usage: run_box.sh <script.py> [args...]}
shift

MAIN_REV=${MAIN_REV:-cc85ddaa9}
MAIN_WT=${MAIN_WT:-/sgl-workspace/wt/cosy-main}
ANALYSIS_WT=${ANALYSIS_WT:-/sgl-workspace/wt/cosyvoice-analysis}
COSYVOICE=${COSYVOICE:-/sgl-workspace/CosyVoice-utilization-20260912}
REPO=${REPO:-/sgl-workspace/sglang-omni}
ANALYSIS_BRANCH=${ANALYSIS_BRANCH:-analysis/cosyvoice-utilization-20260912}

cd "$REPO"
git fetch https://github.com/sgl-project/sglang-omni.git main
[ -d "$MAIN_WT" ] || git worktree add --detach "$MAIN_WT" "$MAIN_REV"
git -C "$MAIN_WT" checkout --detach "$MAIN_REV"
git fetch https://github.com/Ratish1/sglang-omni.git "$ANALYSIS_BRANCH"
ANALYSIS_REV=$(git rev-parse FETCH_HEAD)
[ -d "$ANALYSIS_WT" ] || git worktree add --detach "$ANALYSIS_WT" "$ANALYSIS_REV"
git -C "$ANALYSIS_WT" checkout --detach "$ANALYSIS_REV"

S2="$ANALYSIS_WT/tasks/cosyvoice_utilization_20260912/stage2"
NAME=$(basename "$SCRIPT" .py)
OUT="$ANALYSIS_WT/artifacts/cosyvoice/$NAME-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

cd "$MAIN_WT"
export PYTHONPATH="$MAIN_WT:$S2/../stage0:$S2:$COSYVOICE:$COSYVOICE/third_party/Matcha-TTS"
git rev-parse HEAD > "$OUT/head.txt"
git -C "$ANALYSIS_WT" rev-parse HEAD > "$OUT/analysis_head.txt"
# Every import the scripts need, before the checkpoint load, so a missing one
# costs seconds instead of a booted engine.
python -c "
import sglang_omni, cosyvoice, matcha.utils.audio, sglang, sgl_kernel.flash_attn
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
print(sglang_omni.__file__); print(cosyvoice.__file__); print(sglang.__version__)
" | tee "$OUT/import_path.txt"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv > "$OUT/gpus_before.csv"
nvidia-smi -i 0 --query-compute-apps=pid,used_memory --format=csv
uptime > "$OUT/host_load.txt"; nproc >> "$OUT/host_load.txt"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python "$S2/$SCRIPT" \
  --device cuda:0 --out "$OUT" "$@" 2>&1 | tee "$OUT/run.log"

cd "$ANALYSIS_WT"
tar -czf "$(basename "$OUT").tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
ls -la "$ANALYSIS_WT/$(basename "$OUT").tar.gz"
