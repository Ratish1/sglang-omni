#!/usr/bin/env bash
# One command for a stage 2 box experiment on moss, run from the Mac:
#
#   ssh moss 'docker exec -i sglang-omni-ratish bash -s' < run_box.sh
#   SCRIPT=g0_hop_cache_numerics.py ssh moss 'docker exec -i sglang-omni-ratish bash -s' < run_box.sh
#
# It reuses the container's own repo and venv. It adds one detached worktree for
# the revision under test, because the box's checkout sits on a feature branch,
# and it materialises the experiment scripts as files rather than as a second
# tree. Nothing else is created.
#
#   REPO      container checkout, holds .git and .venv   /workspace/sglang-omni
#   REV       revision under test                        upstream main
#   COSYVOICE CosyVoice clone with its Matcha submodule  /workspace/CosyVoice
#   SCRIPT    experiment to run                          g0_hop_cache_numerics.py
#   ARGS      extra arguments for it                     empty
set -euo pipefail

REPO=${REPO:-/workspace/sglang-omni}
COSYVOICE=${COSYVOICE:-/workspace/CosyVoice}
ANALYSIS_BRANCH=${ANALYSIS_BRANCH:-analysis/cosyvoice-utilization-20260912}
SCRIPT=${SCRIPT:-g0_hop_cache_numerics.py}
ARGS=${ARGS:-}

cd "$REPO"
source .venv/bin/activate
mkdir -p .tmp
grep -qx '.tmp/' .git/info/exclude 2>/dev/null || echo '.tmp/' >> .git/info/exclude

# The revision under test. Objects only; the worktree is the one extra directory.
git fetch --no-tags https://github.com/sgl-project/sglang-omni.git main
REV=${REV:-$(git rev-parse FETCH_HEAD)}
if [ -d .tmp/wt/main ]; then
  git -C .tmp/wt/main checkout --detach "$REV"
else
  git worktree add --detach .tmp/wt/main "$REV"
fi
MAIN="$REPO/.tmp/wt/main"

# The experiment scripts, as files, from the analysis branch.
git fetch --no-tags https://github.com/Ratish1/sglang-omni.git "$ANALYSIS_BRANCH"
ANALYSIS=$(git rev-parse FETCH_HEAD)
T=tasks/cosyvoice_utilization_20260912
mkdir -p .tmp/stage0 .tmp/stage2
for f in stage0/common.py stage2/"$SCRIPT"; do
  git show "$ANALYSIS:$T/$f" > ".tmp/$f"
done

# A card nobody else is on. Never take one that is in use.
CARD=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
       | awk -F', ' '$2 ~ /^0 MiB/ {print $1; exit}')
[ -n "$CARD" ] || { echo "no free card:"; nvidia-smi --query-gpu=index,memory.used --format=csv; exit 1; }

OUT="$REPO/.tmp/out/$(basename "$SCRIPT" .py)-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
echo "$REV" > "$OUT/head.txt"
echo "$ANALYSIS" > "$OUT/analysis_head.txt"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used --format=csv > "$OUT/gpus_before.csv"
uptime > "$OUT/host_load.txt"; nproc >> "$OUT/host_load.txt"

cd "$MAIN"
export PYTHONPATH="$MAIN:$REPO/.tmp/stage0:$REPO/.tmp/stage2:$COSYVOICE:$COSYVOICE/third_party/Matcha-TTS"
python -c "
import sglang_omni, cosyvoice, matcha.utils.audio, sglang, sgl_kernel.flash_attn, common
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT, solve_flow_euler_packed
print(sglang_omni.__file__); print(cosyvoice.__file__)
" | tee "$OUT/import_path.txt"

echo "card $CARD, revision $REV, out $OUT"
CUDA_VISIBLE_DEVICES=$CARD python "$REPO/.tmp/stage2/$SCRIPT" \
  --device cuda:0 --out "$OUT" $ARGS 2>&1 | tee "$OUT/run.log"
echo "OUT=$OUT"
