#!/usr/bin/env bash
# Rebuilds the perf container after a new lease: main tree, the PR trees from patches, the
# Fun-CosyVoice3 dependencies, the kit, and the model and dataset in the cache.
# usage (in the container): bash setup_20261009.sh <dir holding item1.patch and item2.patch>
set -euo pipefail
PATCHES=$1
MAIN=a093c7a2542fc513d0d850ed090bf9d900e945b5
SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')

cd /workspace
git -C sglang-omni fetch -q --depth 1 upstream "$MAIN"
git -C sglang-omni checkout -q "$MAIN"
cd /workspace/sglang-omni
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e ".[fun-cosyvoice3]"
echo "main installed $(git rev-parse --short HEAD)"

apt-get update -qq
apt-get install -y -qq sox > /dev/null

if [ ! -d /workspace/CosyVoice/.git ]; then
  git clone -q https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice
fi
git -C /workspace/CosyVoice checkout -q 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C /workspace/CosyVoice submodule update -q --init third_party/Matcha-TTS
git -C /workspace/CosyVoice/third_party/Matcha-TTS checkout -q dd9105b34bf2be2230f4aa1e4769fb586a3c824e
printf '%s\n' /workspace/CosyVoice /workspace/CosyVoice/third_party/Matcha-TTS > "$SITE/cosyvoice.pth"

git config --global user.name perf
git config --global user.email perf@localhost
for tree in finals3 convk3; do
  if [ ! -d "/workspace/$tree" ]; then
    git -C /workspace/sglang-omni worktree add -q --detach "/workspace/$tree" "$MAIN"
  fi
done
git -C /workspace/finals3 apply "$PATCHES/item1.patch"
git -C /workspace/convk3 apply "$PATCHES/item1.patch"
git -C /workspace/convk3 apply "$PATCHES/item2.patch"
for tree in finals3 convk3; do
  echo "$tree changed=$(git -C /workspace/$tree status --short | wc -l)"
done

if [ ! -d /workspace/tools/.git ]; then
  git clone -q --single-branch -b analysis/cosyvoice-utilization-20260912 \
    https://github.com/Ratish1/sglang-omni.git /workspace/tools
fi
echo "tools $(git -C /workspace/tools log --oneline -1)"

python3 - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
snapshot_download("zhaochenyang20/seed-tts-eval-arrow", repo_type="dataset")
EOF
python3 -c "import sglang, torch, triton, cosyvoice, matcha; print('sglang', sglang.__version__, 'torch', torch.__version__, 'triton', triton.__version__)"
echo "setup done $(date +%T)"
