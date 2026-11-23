#!/usr/bin/env bash
# 11-23 second lease (node-0001, 2 GPUs, 6 h): the two reads lost when the first lease ended. Trees: upstream main
# 22bf41ef5, the full prefill graph on it 24ef9b82c, part 2's final head F 085064564 and its three beside-video arms.
set -eu
O=/data/v1123b
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "setup start"
cd /workspace/sglang-omni
git fetch -q upstream 22bf41ef53a2565fcca65fc2b7b29e056fba1310
git fetch -q origin dev/qwen3-omni-full-prefill-on-main dev/qwen3-omni-one-context-visual-graph-read \
  dev/qwen3-omni-oc-bisect-blocking-copies dev/qwen3-omni-oc-bisect-no-stream-priority dev/qwen3-omni-oc-bisect-read-in-turn
git worktree add -q --detach /workspace/umain 22bf41ef53a2565fcca65fc2b7b29e056fba1310
git worktree add -q --detach /workspace/fullmain origin/dev/qwen3-omni-full-prefill-on-main
git worktree add -q --detach /workspace/ocvg7 08506456426230423264f4d09ec0df80353fc465
for arm in blocking-copies no-stream-priority read-in-turn; do
  git worktree add -q --detach /workspace/x-$arm origin/dev/qwen3-omni-oc-bisect-$arm
done
git fetch -q origin analysis/runtime-perf-research && git worktree add -q --detach /workspace/kit FETCH_HEAD
for tree in umain fullmain ocvg7 x-blocking-copies x-no-stream-priority x-read-in-turn kit; do
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
snapshot_download('zhaochenyang20/seed-tts-eval-arrow', repo_type='dataset')
snapshot_download('zhaochenyang20/Video_MME_ci', repo_type='dataset')
"
log "setup done"
