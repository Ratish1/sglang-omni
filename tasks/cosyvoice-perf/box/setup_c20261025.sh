#!/usr/bin/env bash
# 10-25, a new lease on host-85-234-79-221 (4 H100, cards 0 and 1 only), the 0.5.22 container
# sglang-omni-cosyvoice-0522 on CI's image hongccc/sglang-omni@sha256:06ed3067 (the 10-23 digest): sox, the editable
# tree at #3129's merge base cd191df23 with the fun-cosyvoice3 extra, CosyVoice and Matcha-TTS at the cookbook's commits
# on a .pth, the checkpoint, the trace marks worktree, the arms t_mn (the merge base), t_l4t (#3129 with the rereview
# fixes, from the bundle at /data/c20261025/l4t/l4t.bundle) and t_l4o (#3129's reviewed head adc65b396, for the new
# test's pre-fix run).
# usage (in the container): mkdir -p /data/c20261025 && bash setup_c20261025.sh > /data/c20261025/setup.log 2>&1
set -x
MAIN=cd191df237f8c21e631133c6cf349164e6c1fcdf
OLD=adc65b3962f0af19bf1bc7466e5defca545044ef
apt-get update -qq && apt-get install -y -qq sox > /dev/null
cd /workspace/sglang-omni
git fetch -q upstream $MAIN
git checkout -q --detach $MAIN && git log --oneline -1
git fetch -q /data/c20261025/l4t/l4t.bundle perf/cosyvoice3-prefix-pool-headroom
L4T=$(git rev-parse FETCH_HEAD)
git cat-file -e $OLD^{commit}
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e ".[fun-cosyvoice3]"
[ -d /workspace/CosyVoice ] || git clone -q --recursive https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice
git -C /workspace/CosyVoice checkout -q 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C /workspace/CosyVoice submodule update -q --init --recursive
git -C /workspace/CosyVoice/third_party/Matcha-TTS checkout -q dd9105b34bf2be2230f4aa1e4769fb586a3c824e
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
printf '/workspace/CosyVoice\n/workspace/CosyVoice/third_party/Matcha-TTS\n' > "$SITE/cosyvoice_paths.pth"
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512'))"
git fetch -q origin prof/cosyvoice3-trace-marks
[ -d /workspace/t_marks ] || git worktree add -q --detach /workspace/t_marks FETCH_HEAD
[ -d /workspace/t_mn ] || git worktree add -q --detach /workspace/t_mn $MAIN
[ -d /workspace/t_l4t ] || git worktree add -q --detach /workspace/t_l4t $L4T
[ -d /workspace/t_l4o ] || git worktree add -q --detach /workspace/t_l4o $OLD
for tree in t_marks t_mn t_l4t t_l4o; do
  echo "$tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-60) $(git -C /workspace/$tree rev-parse HEAD:sglang_omni) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
python3 -c "import sglang, torch, flashinfer, cosyvoice, onnxruntime; print('sglang', sglang.__version__, 'torch', torch.__version__, 'cudnn', torch.backends.cudnn.version(), 'flashinfer', flashinfer.__version__, 'onnxruntime', onnxruntime.__version__)"
echo SETUP_DONE
