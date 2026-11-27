#!/usr/bin/env bash
# 11-28, node-0001, 2 H200s (container GPUs 0 at 08:00.0 and 1 at 7E:00.0, both NUMA 0, so card 0 serves from CPUs
# 0-55,112-167 and card 1 from 56-111,168-223), 10 h: #3452's no-regression bisect. Trees: upstream main 0f70d9c7c (the
# main every 11-27 cell ran against) and the PR head 91c2e2010 (source tree 8c4758c1658d, the tested one).
set -eu
O=/data/v1128
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "setup start"
cd /workspace/sglang-omni
git fetch -q upstream 0f70d9c7c41c3e239dc129cb7b2c321e815c92e4
git fetch -q origin perf/qwen3-omni-preprocessing-workers-bounded
git worktree add -q --detach /workspace/main 0f70d9c7c41c3e239dc129cb7b2c321e815c92e4
git worktree add -q --detach /workspace/fix 91c2e2010bed
git fetch -q origin analysis/runtime-perf-research && git worktree add -q --detach /workspace/kit FETCH_HEAD
for tree in main fix kit; do
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
snapshot_download('zhaochenyang20/seed-tts-eval-arrow', repo_type='dataset')
snapshot_download('zhaochenyang20/Video_MME_ci', repo_type='dataset')
"
log "setup done"
