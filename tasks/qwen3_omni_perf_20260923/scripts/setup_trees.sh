#!/usr/bin/env bash
# Build the run trees of a container under /workspace/sglang-omni/.tmp/wt/, one detached
# worktree per name=remote/branch argument (remote is origin or upstream). An existing tree of
# that name is moved to the new head. Prints each tree's head and the installed versions, so a
# fresh container is set up in one call and its provenance lands in the log.
# usage: setup_trees.sh main=upstream/main tools=origin/feat/omni-step-profiler ...
set -euo pipefail
cd /workspace/sglang-omni
mkdir -p .tmp/wt
for spec in "$@"; do
  name=${spec%%=*} ref=${spec#*=}
  remote=${ref%%/*} branch=${ref#*/}
  git fetch -q "$remote" "$branch"
  #FETCH_HEAD belongs to this checkout, so the tree gets the resolved commit
  sha=$(git rev-parse FETCH_HEAD)
  if [ -d ".tmp/wt/$name" ]; then
    git -C ".tmp/wt/$name" checkout -q --detach "$sha"
  else
    git worktree add -q --detach ".tmp/wt/$name" "$sha"
  fi
  echo "$name $(git -C ".tmp/wt/$name" log --oneline -1) ($ref)"
done
python3 -c "import sglang, torch, transformers; print('sglang', sglang.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
