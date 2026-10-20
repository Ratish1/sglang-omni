#!/usr/bin/env bash
# Rebuilds the perf container on the 10-11 lease (2 H100, cards 0 and 1): main cf496151d with the
# Fun-CosyVoice3 dependencies, CosyVoice and Matcha-TTS at their pinned commits, the kit, the model and the
# SeedTTS set in the cache, the long form list, and the trees for item 3: main, prerequisites 1 and 2, and
# prerequisites 1 and 2 with item 3 v3.
# usage (in the container): bash setup_c20261011.sh <dir holding main_cf496151d_20261010/>
set -euo pipefail
PATCHES=$1/main_cf496151d_20261010
MAIN=cf496151dfa6eed47e7ed466e451a6e3eb1204d6
R=/data/c20261011
SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
mkdir -p $R

git -C /workspace/sglang-omni fetch -q upstream main
git -C /workspace/sglang-omni checkout -q "$MAIN"
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
add() {
  local tree=$1
  shift
  if [ -d "/workspace/$tree" ]; then
    echo "$tree exists"
    exit 1
  else
    git -C /workspace/sglang-omni worktree add -q --detach "/workspace/$tree" "$MAIN"
  fi
  for patch in "$@"; do
    git -C "/workspace/$tree" apply -3 "$PATCHES/$patch.patch"
    git -C "/workspace/$tree" add -A
  done
  git -C "/workspace/$tree" reset -q
  echo "$tree on $MAIN with $* changed=$(git -C "/workspace/$tree" status --short | wc -l)" \
    | tee "/workspace/$tree/BRANCH_MARKER.txt"
}
add t_main
add t_p12 p1_tokenizer_arena p2_expandable_segments
add t_p123v3 stack_p1_p2_i3v3

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
KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
PYTHONPATH=/workspace/sglang-omni python3 $KIT/make_longform_meta.py --out $R/longform
wc -l $R/longform/meta.lst
python3 -c "import sglang, torch, triton, onnxruntime, cosyvoice, matcha; print('sglang', sglang.__version__, 'torch', torch.__version__, 'triton', triton.__version__, 'onnxruntime', onnxruntime.__version__)"
echo "setup done $(date +%T)"
