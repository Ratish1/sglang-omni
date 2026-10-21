#!/usr/bin/env bash
# 10-21, the 0.5.22 container (sglang-omni-cosyvoice-0522, CI's image hongccc/sglang-omni@sha256:06ed3067, cards 0
# and 1): sox, the editable tree at upstream main 057d97629 with the fun-cosyvoice3 extra, CosyVoice and Matcha-TTS
# at the pinned commits on a .pth, the checkpoint (the node's shared HF cache), then the worktrees t_main at main and
# t_h0 at #3029's head before this round, runs under /data/c20261021b.
# usage (in the container): bash setup_c20261021b.sh > /data/c20261021b/setup.log 2>&1
set -x
MAIN=057d97629da4093ea7b02ae9487803cf27f88749
H0=fd2139ac6
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
[ -d /workspace/t_main ] || git worktree add -q --detach /workspace/t_main $MAIN
git fetch -q origin perf/cosyvoice3-hift-channels-last
[ -d /workspace/t_h0 ] || git worktree add -q --detach /workspace/t_h0 $H0
for tree in t_main t_h0; do git -C /workspace/$tree log --oneline -1; git -C /workspace/$tree rev-parse HEAD:sglang_omni; done
python3 -c "import sglang, torch, flashinfer, cosyvoice; print('sglang', sglang.__version__, 'torch', torch.__version__, 'cudnn', torch.backends.cudnn.version(), 'flashinfer', flashinfer.__version__)"
echo SETUP_DONE
