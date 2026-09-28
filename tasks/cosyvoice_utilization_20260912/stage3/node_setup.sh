#!/usr/bin/env bash
# Fresh radix lease: install the fun-cosyvoice3 extra, the pinned CosyVoice checkout on a .pth,
# and the analysis worktree. Idempotent.
#   bash node_setup.sh > /data/setup.log 2>&1
set -euxo pipefail
cd /workspace/sglang-omni
git fetch -q upstream main && git checkout -q FETCH_HEAD
command -v sox || (apt-get update -qq && apt-get install -y -qq sox)
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e ".[fun-cosyvoice3]"
COSYVOICE=/data/src/CosyVoice
if [ ! -d "$COSYVOICE" ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE"
  git -C "$COSYVOICE" checkout -q --detach 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
  git -C "$COSYVOICE" submodule update -q --init --depth=1 third_party/Matcha-TTS
else
  :
fi
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
printf '%s\n' "$COSYVOICE" "$COSYVOICE/third_party/Matcha-TTS" > "$SITE/cosyvoice.pth"
mkdir -p .tmp/logs
git fetch -q origin analysis/cosyvoice-utilization-20260912
ANALYSIS=$(git rev-parse FETCH_HEAD)
if [ -d .tmp/wt/analysis ]; then
  git -C .tmp/wt/analysis checkout -q --detach "$ANALYSIS"
else
  git worktree add -q --detach .tmp/wt/analysis "$ANALYSIS"
fi
python3 -c "import torch, sglang, cosyvoice.flow.DiT.dit; print(torch.__version__, sglang.__version__)"
echo SETUP_DONE
