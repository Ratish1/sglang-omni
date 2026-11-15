#!/usr/bin/env bash
# 11-15, node-0000, two H200 (card 0 NUMA 0, card 1 NUMA 1): worktrees by hash, install, model snapshot.
set -eu
O=/data/rp1115
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "setup start"
cd /workspace/sglang-omni
git fetch -q origin 35890f10c423d768cb267446100cfda75bcfe426 053a02a6f1cf071ba7d0a105bb3ae3448d37ca5f \
  7342cfcff2a295ed633d217bbaa37cdfb6eca977
git worktree add -q --detach /workspace/main 35890f10c423d768cb267446100cfda75bcfe426
git worktree add -q --detach /workspace/fix 053a02a6f1cf071ba7d0a105bb3ae3448d37ca5f
git worktree add -q --detach /workspace/fold 7342cfcff2a295ed633d217bbaa37cdfb6eca977
git fetch -q origin analysis/runtime-perf-research && git worktree add -q --detach /workspace/kit2 FETCH_HEAD
git fetch -q origin analysis/qwen3-omni-perf-20260923 && git worktree add -q --detach /workspace/marks FETCH_HEAD
for tree in main fix fold kit2 marks; do
  log "tree $tree $(git -C /workspace/$tree rev-parse --short HEAD) $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni 2>/dev/null)"
done
cd /workspace/fold
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e .
cubins=$(uv pip list --python "$(command -v python3)" 2>/dev/null | awk '/^flashinfer-(cubin|jit-cache)/ {print $1}')
if [ -n "$cubins" ]; then uv pip uninstall -q --python "$(command -v python3)" $cubins; else :; fi
uv pip install -q --python "$(command -v python3)" nvtx
log "installed $(uv pip list --python "$(command -v python3)" 2>/dev/null | grep -E '^(sglang|flashinfer|torch|nvtx) ' | tr -s ' ' | tr '\n' ';')"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct')"
log "setup done"
