#!/usr/bin/env bash
# 10-25 evening, a new lease on host-85-234-79-221 (4 H100, cards 0 and 1 only) after the 17:26 expiry: the 0.5.22
# container sglang-omni-cosyvoice-0522 on CI's image hongccc/sglang-omni@sha256:06ed3067; sox, the editable tree at main
# ab8193645 (#3132's landing base) with the fun-cosyvoice3 extra, CosyVoice and Matcha-TTS at the cookbook's commits on
# a .pth, the checkpoint, the speaker similarity assets (fetched once, so two scorers never race on a cold cache), the
# trace marks worktree, the arms t_mq (main ab8193645) and t_l6q (#3132's head, origin).
# usage (in the container): mkdir -p /data/c20261025b && bash setup_c20261025b.sh > /data/c20261025b/setup.log 2>&1
set -x
MAIN=ab81936454eaec8d8b8b0e31446f5ee563020f0d
apt-get update -qq && apt-get install -y -qq sox > /dev/null
cd /workspace/sglang-omni
git fetch -q upstream $MAIN
git checkout -q --detach $MAIN && git log --oneline -1
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e ".[fun-cosyvoice3]"
[ -d /workspace/CosyVoice ] || git clone -q --recursive https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice
git -C /workspace/CosyVoice checkout -q 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
git -C /workspace/CosyVoice submodule update -q --init --recursive
git -C /workspace/CosyVoice/third_party/Matcha-TTS checkout -q dd9105b34bf2be2230f4aa1e4769fb586a3c824e
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
printf '/workspace/CosyVoice\n/workspace/CosyVoice/third_party/Matcha-TTS\n' > "$SITE/cosyvoice_paths.pth"
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512'))"
python3 -c "from benchmarks.metrics.speaker_similarity_assets import ensure_speaker_similarity_assets; print(ensure_speaker_similarity_assets())"
git fetch -q origin prof/cosyvoice3-trace-marks
[ -d /workspace/t_marks ] || git worktree add -q --detach /workspace/t_marks FETCH_HEAD
git fetch -q origin perf/cosyvoice3-prefix-graph-ladder-v2
[ -d /workspace/t_l6q ] || git worktree add -q --detach /workspace/t_l6q FETCH_HEAD
[ -d /workspace/t_mq ] || git worktree add -q --detach /workspace/t_mq $MAIN
for tree in t_marks t_mq t_l6q; do
  echo "$tree $(git -C /workspace/$tree log --oneline -1 | cut -c1-60) $(git -C /workspace/$tree rev-parse HEAD:sglang_omni) $(git -C /workspace/$tree diff --quiet HEAD -- sglang_omni || echo DIRTY)"
done
python3 -c "import sglang, torch, flashinfer, cosyvoice, onnxruntime; print('sglang', sglang.__version__, 'torch', torch.__version__, 'cudnn', torch.backends.cudnn.version(), 'flashinfer', flashinfer.__version__, 'onnxruntime', onnxruntime.__version__)"
echo SETUP_DONE
