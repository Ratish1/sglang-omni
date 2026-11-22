#!/usr/bin/env bash
# 11-23 lease (node-0001, GPUs 1 and 3, 10 h): worktrees by hash, install, model and dataset snapshots.
# Trees: upstream main 22bf41ef5 (fix A's landing base), fix A merged on it 3491dc8a7, part 2's base 6aff09f46,
# part 2's final combined head 085064564.
set -eu
O=/data/v1123
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "setup start"
cd /workspace/sglang-omni
UMAIN=22bf41ef53a2565fcca65fc2b7b29e056fba1310
FIXA=3491dc8a79144b087d4f718f31fc0290934c4cb8
BVG=6aff09f462317ca7bed4887b52b3035383becf0b
OCVG7=08506456426230423264f4d09ec0df80353fc465
git fetch -q upstream $UMAIN
git fetch -q origin perf/thinker-index-copies-without-stream-wait dev/qwen3-omni-both-changes-visual-graph \
  dev/qwen3-omni-one-context-visual-graph-read
git worktree add -q --detach /workspace/umain $UMAIN
git worktree add -q --detach /workspace/fixa $FIXA
git worktree add -q --detach /workspace/bvg $BVG
git worktree add -q --detach /workspace/ocvg7 $OCVG7
git fetch -q origin analysis/runtime-perf-research && git worktree add -q --detach /workspace/kit FETCH_HEAD
for tree in umain fixa bvg ocvg7 kit; do
  log "tree $tree $(git -C /workspace/$tree rev-parse --short HEAD) $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni 2>/dev/null)"
done
cd /workspace/umain
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e .
cubins=$(uv pip list --python "$(command -v python3)" 2>/dev/null | awk '/^flashinfer-(cubin|jit-cache)/ {print $1}')
if [ -n "$cubins" ]; then uv pip uninstall -q --python "$(command -v python3)" $cubins; else :; fi
uv pip install -q --python "$(command -v python3)" nvtx
log "installed $(uv pip list --python "$(command -v python3)" 2>/dev/null | grep -E '^(sglang|flashinfer|torch|nvtx) ' | tr -s ' ' | tr '\n' ';')"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct')
snapshot_download('Qwen/Qwen3-ASR-1.7B')
snapshot_download('zhaochenyang20/seed-tts-eval-arrow', repo_type='dataset')
snapshot_download('zhaochenyang20/Video_MME_ci', repo_type='dataset')
"
log "setup done"
