#!/usr/bin/env bash
# Run a stage 2 experiment on the moss box, from the Mac:
#
#   ssh moss 'docker exec -i sglang-omni-ratish bash -s' < run_box.sh
#
# Two worktrees off the container's own repo, its own .venv, one free card.
# Nothing is patched: the tree under test is upstream main as published.
#
#   REPO       container checkout, holds .git and .venv  /workspace/sglang-omni
#   REV        revision under test                       upstream main
#   COSYVOICE  CosyVoice clone with its Matcha submodule /workspace/CosyVoice
#   CARDS      cards this box lets us use                0 1 2 3
#   CARD       one card, taken even if shared. Numerics do not care who else
#              is on the card; timings do, so never set this for a timing run.
#   NEED       MiB this run has to fit in, when every card is shared    9000
#   MODEL      checkpoint, a local directory             /data/ms/.../master
#   SCRIPT     experiment to run                         g0_hop_cache_numerics.py
#   ARGS       extra arguments for it                    empty
set -euo pipefail

REPO=${REPO:-/workspace/sglang-omni}
COSYVOICE=${COSYVOICE:-/workspace/CosyVoice}
ANALYSIS_BRANCH=${ANALYSIS_BRANCH:-analysis/cosyvoice-utilization-20260912}
CARDS=${CARDS:-"0 1 2 3"}
# The 9.1 GiB checkpoint lives on the data disk. resolve_checkpoint takes a
# directory as is, so a run never touches the hub for it. Pulled from ModelScope:
# the container's proxy stalls on large HuggingFace files and does not resume.
MODEL=${MODEL:-/data/ms/models/FunAudioLLM--Fun-CosyVoice3-0.5B-2512/snapshots/master}
# A run reads the checkpoint and the cached SeedTTS arrow from disk. Offline so
# it cannot silently reach for the network and stall there instead of measuring.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
SCRIPT=${SCRIPT:-g0_hop_cache_numerics.py}
ARGS=${ARGS:-}

# The container exports both a SOCKS and an HTTP proxy; httpx prefers SOCKS and
# then wants socksio, which is not installed. The HTTP proxy is enough.
unset ALL_PROXY all_proxy

cd "$REPO"
source .venv/bin/activate
mkdir -p .tmp
grep -qx '.tmp/' .git/info/exclude 2>/dev/null || echo '.tmp/' >> .git/info/exclude

worktree() {  # name, revision
  if [ -d ".tmp/wt/$1" ]; then
    git -C ".tmp/wt/$1" checkout --detach "$2"
  else
    git worktree add --detach ".tmp/wt/$1" "$2"
  fi
}

git fetch --no-tags https://github.com/sgl-project/sglang-omni.git main
REV=${REV:-$(git rev-parse FETCH_HEAD)}
worktree main "$REV"

git fetch --no-tags https://github.com/Ratish1/sglang-omni.git "$ANALYSIS_BRANCH"
ANALYSIS=$(git rev-parse FETCH_HEAD)
worktree analysis "$ANALYSIS"

MAIN="$REPO/.tmp/wt/main"
T="$REPO/.tmp/wt/analysis/tasks/cosyvoice_utilization_20260912"

# A card nobody else is on, from the ones this box lets us use. Idle cards report
# 1 MiB here, so ownership is decided by compute processes, not by memory.
if [ -n "${CARD:-}" ]; then
  echo "card $CARD taken explicitly; free memory on it:"
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader -i "$CARD"
else
  # An idle card if there is one. Otherwise the one with the most memory free:
  # the box is shared and every card carries someone, so refusing outright just
  # means never running. NEED is what this run has to fit in.
  NEED=${NEED:-9000}
  BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  CARD=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
    | awk -F', ' -v busy="$BUSY" -v cards=" $CARDS " '
        index(cards, " " $1 " ") && index(busy, $2) == 0 { print $1; exit }')
  if [ -z "$CARD" ]; then
    CARD=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader \
      | awk -F', ' -v cards=" $CARDS " -v need="$NEED" '
          { gsub(" MiB", "", $2); gsub(" MiB", "", $3); free = $2 - $3 }
          index(cards, " " $1 " ") && free > best && free > need { best = free; pick = $1 }
          END { if (pick != "") print pick }')
    [ -n "$CARD" ] && echo "no idle card; taking $CARD, the one with the most free memory"
  fi
  [ -n "$CARD" ] || { echo "no card among $CARDS has ${NEED} MiB free:"; nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv; exit 1; }
fi

OUT="$REPO/.tmp/out/$(basename "$SCRIPT" .py)-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
echo "$REV" > "$OUT/head.txt"
echo "$ANALYSIS" > "$OUT/analysis_head.txt"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used --format=csv > "$OUT/gpus_before.csv"
uptime > "$OUT/host_load.txt"; nproc >> "$OUT/host_load.txt"

cd "$MAIN"
export PYTHONPATH="$MAIN:$T/stage0:$T/stage2:$COSYVOICE:$COSYVOICE/third_party/Matcha-TTS"
python -c "
import sglang_omni, cosyvoice, matcha.utils.audio, sglang, sgl_kernel.flash_attn, common
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT, solve_flow_euler_packed
print(sglang_omni.__file__); print(cosyvoice.__file__)
" | tee "$OUT/import_path.txt"

echo "card $CARD, revision $REV, model $MODEL"
echo "out $OUT"
# -u so the log follows the run instead of arriving at the end.
CUDA_VISIBLE_DEVICES=$CARD python -u "$T/stage2/$SCRIPT" \
  --device cuda:0 --model "$MODEL" --out "$OUT" $ARGS 2>&1 | tee "$OUT/run.log"
echo "OUT=$OUT"
