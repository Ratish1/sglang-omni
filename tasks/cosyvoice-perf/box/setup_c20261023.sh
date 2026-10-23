#!/usr/bin/env bash
# 10-23, a new lease on host-85-234-79-221 (the 10-22 home went with the old lease), the 0.5.22 container
# sglang-omni-cosyvoice-0522 on CI's image hongccc/sglang-omni@sha256:06ed3067 (the 10-22 digest), cards 0 and 1:
# sox, the editable tree at main 1b726ab64 with the fun-cosyvoice3 extra, CosyVoice and Matcha-TTS at the cookbook's
# commits on a .pth, the checkpoint, the trace marks worktree and the arms t_m1b, t_chain, t_l4b and t_l5b.
# usage (in the container): mkdir -p /data/c20261022b && bash setup_c20261023.sh > /data/c20261022b/setup_c20261023.log 2>&1
set -x
MAIN=1b726ab64680661e484196f80485f4e6062e5afa
CHAIN=fdf92c2e52e279d4b6e4bff8aae7a7cdd97baf6f
L4B=4f3ba8bd9dace323a7f4a15e16b98e3e033063b5
L5B=4eb6e93e7a5bcb20b4c3f34ee85c110d0b42d4e3
apt-get update -qq && apt-get install -y -qq sox > /dev/null
cd /workspace/sglang-omni
git fetch -q upstream $MAIN
git fetch -q origin perf/cosyvoice3-items-chain
git cat-file -e $CHAIN^{commit} && git cat-file -e $L4B^{commit} && git cat-file -e $L5B^{commit}
git checkout -q --detach $MAIN && git log --oneline -1
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
[ -d /workspace/t_m1b ] || git worktree add -q --detach /workspace/t_m1b $MAIN
[ -d /workspace/t_chain ] || git worktree add -q --detach /workspace/t_chain $CHAIN
[ -d /workspace/t_l4b ] || git worktree add -q --detach /workspace/t_l4b $L4B
[ -d /workspace/t_l5b ] || git worktree add -q --detach /workspace/t_l5b $L5B
for tree in t_marks t_m1b t_chain t_l4b t_l5b; do
  echo "$tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-60) $(git -C /workspace/$tree rev-parse HEAD:sglang_omni) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
python3 -c "import sglang, torch, flashinfer, cosyvoice, onnxruntime; print('sglang', sglang.__version__, 'torch', torch.__version__, 'cudnn', torch.backends.cudnn.version(), 'flashinfer', flashinfer.__version__, 'onnxruntime', onnxruntime.__version__)"
echo SETUP_DONE
