#!/usr/bin/env bash
# 11-22, node-0001, two H200 (GPUs 1 and 3, the same cards as 11-21): worktrees by hash, install, model and dataset
# snapshots. Trees: main e0482a08c, part 2's base 6aff09f46, combined 7cae415a1, full stack 9ca07fb99.
set -eu
O=/data/v1122
mkdir -p $O
log() { echo "$* $(date -u +%T)" >> $O/queue.log; }
log "setup start"
cd /workspace/sglang-omni
NMAIN=e0482a08cf48e6333952c974f3a5e3a4a3398361
BVG=6aff09f462317ca7bed4887b52b3035383becf0b
OCVG=7cae415a1b0e67e9d150b8e43df4936faaa72b31
SOCVG=9ca07fb993f6aaefd719f0c76f8b23b686f984ce
git fetch -q upstream $NMAIN
git fetch -q origin dev/qwen3-omni-both-changes-visual-graph dev/qwen3-omni-one-context-visual-graph-read \
  dev/qwen3-omni-stack-one-context-visual-graph
git worktree add -q --detach /workspace/nmain $NMAIN
git worktree add -q --detach /workspace/bvg $BVG
git worktree add -q --detach /workspace/ocvg $OCVG
git worktree add -q --detach /workspace/socvg $SOCVG
git fetch -q origin analysis/runtime-perf-research && git worktree add -q --detach /workspace/kit FETCH_HEAD
for tree in nmain bvg ocvg socvg kit; do
  log "tree $tree $(git -C /workspace/$tree rev-parse --short HEAD) $(git -C /workspace/$tree rev-parse --short=12 HEAD:sglang_omni 2>/dev/null)"
done
cd /workspace/nmain
uv pip install -q --python "$(command -v python3)" --prerelease=allow -e .
cubins=$(uv pip list --python "$(command -v python3)" 2>/dev/null | awk '/^flashinfer-(cubin|jit-cache)/ {print $1}')
if [ -n "$cubins" ]; then uv pip uninstall -q --python "$(command -v python3)" $cubins; else :; fi
uv pip install -q --python "$(command -v python3)" nvtx
log "installed $(uv pip list --python "$(command -v python3)" 2>/dev/null | grep -E '^(sglang|flashinfer|torch|nvtx) ' | tr -s ' ' | tr '\n' ';')"
for card in 0 1; do
  log "card $card numa $(cat /sys/bus/pci/devices/$(nvidia-smi -i $card --query-gpu=pci.bus_id --format=csv,noheader | tr 'A-F' 'a-f' | sed 's/^0000\(0000\)*/0000/')/numa_node 2>/dev/null)"
done
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct')
snapshot_download('Qwen/Qwen3-ASR-1.7B')
snapshot_download('zhaochenyang20/seed-tts-eval-arrow', repo_type='dataset')
snapshot_download('zhaochenyang20/Video_MME_ci', repo_type='dataset')
"
log "setup done"
