#!/usr/bin/env bash
# 10-20 lease (host-85-234-79-62, cards 2 and 3 of our 4, a fresh container: the new lease wiped the home): the
# install (sox, the editable tree at upstream main with the fun-cosyvoice3 extra, CosyVoice and Matcha-TTS at the
# pinned commits on a .pth, the checkpoint), then the worktree t_m20 at upstream main b98a28936.
# usage (in the container): bash setup_c20261020.sh > /data/c20261020/setup.log 2>&1
set -x
MAIN=b98a28936
apt-get update -qq && apt-get install -y -qq sox > /dev/null
cd /workspace/sglang-omni
git cat-file -e $MAIN^{commit} 2> /dev/null || git fetch -q --depth 50 upstream main
git checkout -q --detach $MAIN && git log --oneline -1
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e ".[fun-cosyvoice3]"
[ -d /workspace/CosyVoice ] || git clone -q --recursive https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice
git -C /workspace/CosyVoice checkout -q 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C /workspace/CosyVoice submodule update -q --init --recursive
git -C /workspace/CosyVoice/third_party/Matcha-TTS checkout -q dd9105b34bf2be2230f4aa1e4769fb586a3c824e
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
printf '/workspace/CosyVoice\n/workspace/CosyVoice/third_party/Matcha-TTS\n' > "$SITE/cosyvoice_paths.pth"
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512'))"
[ -d /workspace/t_m20 ] || git worktree add -q --detach /workspace/t_m20 $MAIN
for tree in t_m20; do git -C /workspace/$tree log --oneline -1; git -C /workspace/$tree rev-parse HEAD:sglang_omni; done
python3 -c "import sglang, torch, flashinfer, cosyvoice; print('sglang', sglang.__version__, 'torch', torch.__version__, 'flashinfer', flashinfer.__version__)"
echo SETUP_DONE
