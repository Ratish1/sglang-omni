#!/usr/bin/env bash
# Trees for the 10-10 verification of the Fun-CosyVoice3 item branches ported onto main cf496151d:
# each branch alone, the stack (prerequisites 1 and 2, items 3 and 4) and the stack with item 5,
# the kit, and the long form lists (48 x 10 texts, and 32 x 40 texts for the length stress).
# usage (in the container): bash setup_c20261010.sh <dir holding the patches>
set -euo pipefail
PATCHES=$1
MAIN=cf496151dfa6eed47e7ed466e451a6e3eb1204d6
R=/data/c20261010
mkdir -p $R
[ "$(git -C /workspace/sglang-omni rev-parse HEAD)" = "$MAIN" ] || { echo "main tree is not at $MAIN"; exit 1; }
git config --global user.name perf
git config --global user.email perf@localhost

add() {
  local tree=$1
  shift
  if [ -d "/workspace/$tree" ]; then
    echo "$tree exists"
    exit 1
  else
    git -C /workspace/sglang-omni worktree add -q --detach "/workspace/$tree" "$MAIN"
  fi
  for patch in "$@"; do
    git -C "/workspace/$tree" apply -3 "$PATCHES/$patch.patch"
    git -C "/workspace/$tree" add -A
  done
  git -C "/workspace/$tree" reset -q
  echo "$tree on $MAIN with $* changed=$(git -C "/workspace/$tree" status --short | wc -l)" \
    | tee "/workspace/$tree/BRANCH_MARKER.txt"
}

add t_p1 p1_tokenizer_arena
add t_p2 p2_expandable_segments
add t_i3 i3_prefix_pool_free_memory
add t_i4 i4_one_step_both_kinds
add t_i5 i5_ar_breakable_prefill
add t_stack stack_p1_p2_i3_i4
add t_stack5 stack_p1_p2_i3_i4 i5_ar_breakable_prefill

if [ ! -d /workspace/tools/.git ]; then
  git clone -q --single-branch -b analysis/cosyvoice-utilization-20260912 \
    https://github.com/Ratish1/sglang-omni.git /workspace/tools
else
  :
fi
echo "tools $(git -C /workspace/tools log --oneline -1)"

KIT=/workspace/tools/tasks/cosyvoice_utilization_20260912/stage4
cd /workspace/sglang-omni
PYTHONPATH=/workspace/sglang-omni python3 $KIT/make_longform_meta.py --out $R/longform
PYTHONPATH=/workspace/sglang-omni python3 $KIT/make_longform_meta.py --out $R/stress \
  --entries 32 --texts-per-entry 40
wc -l $R/longform/meta.lst $R/stress/meta.lst
echo "setup done $(date +%T)"
