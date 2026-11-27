#!/usr/bin/env bash
# 11-27, node-0000, the one H200 at host index 1 (the container sees it as GPU 0): #3452's review fixes. Trees: upstream
# main 0f70d9c7c, the PR head 4b1d6b482 and the fixes 72be1f80e (main merged into the PR head, then the five fixes).
set -eu
O=/data/v1127
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "setup start"
cd /workspace/sglang-omni
git fetch -q upstream 0f70d9c7c41c3e239dc129cb7b2c321e815c92e4
git fetch -q origin perf/qwen3-omni-preprocessing-workers-bounded dev/qwen3-omni-video-lane-review-fixes
git worktree add -q --detach /workspace/main 0f70d9c7c41c3e239dc129cb7b2c321e815c92e4
git worktree add -q --detach /workspace/head 4b1d6b4828ce955a8e7d5f8a6b81bf8de4848bbe
git worktree add -q --detach /workspace/fix origin/dev/qwen3-omni-video-lane-review-fixes
git fetch -q origin analysis/runtime-perf-research && git worktree add -q --detach /workspace/kit FETCH_HEAD
for tree in main head fix kit; do
  log "tree $tree $(git -C /workspace/$tree rev-parse --short HEAD) $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni 2>/dev/null)"
done
cd /workspace/main
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e .
cubins=$(uv pip list --python "$(command -v python3)" 2>/dev/null | awk '/^flashinfer-(cubin|jit-cache)/ {print $1}')
if [ -n "$cubins" ]; then uv pip uninstall -q --python "$(command -v python3)" $cubins; else :; fi
log "installed $(uv pip list --python "$(command -v python3)" 2>/dev/null | grep -E '^(sglang|flashinfer|torch|torchcodec|transformers) ' | tr -s ' ' | tr '\n' ';')"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct')
snapshot_download('marksverdhei/Qwen3-Omni-30B-A3B-FP8')
snapshot_download('zhaochenyang20/seed-tts-eval-arrow', repo_type='dataset')
snapshot_download('zhaochenyang20/Video_MME_ci', repo_type='dataset')
snapshot_download('zhaochenyang20/Video_AMME_ci', repo_type='dataset')
"
log "setup done"
