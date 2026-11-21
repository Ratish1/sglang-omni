#!/usr/bin/env bash
# 11-21, node-0001, two H200 (both cards on NUMA 0): worktrees by hash, install, model and dataset snapshots.
set -eu
O=/data/v1121
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "setup start"
cd /workspace/sglang-omni
MAIN=de4cb9f0fbc5921558daa1073305adcdc67fa81e
STACK=6149c714fbee77c41c95a8f6b24e51d28368b22f
V3452=a18f40fa73f970b4732f19c2f1b8fbb732f05d5e
NMAIN=e0482a08cf48e6333952c974f3a5e3a4a3398361
git fetch -q origin $MAIN $STACK $V3452
git fetch -q upstream $NMAIN
git worktree add -q --detach /workspace/tmain $MAIN
git worktree add -q --detach /workspace/tstack $STACK
git worktree add -q --detach /workspace/v3452 $V3452
git worktree add -q --detach /workspace/nmain $NMAIN
git fetch -q origin analysis/runtime-perf-research && git worktree add -q --detach /workspace/kit FETCH_HEAD
for tree in tmain tstack v3452 nmain kit; do
  log "tree $tree $(git -C /workspace/$tree rev-parse --short HEAD) $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni 2>/dev/null)"
done
cd /workspace/nmain
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
