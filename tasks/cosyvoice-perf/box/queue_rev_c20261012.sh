#!/usr/bin/env bash
# The reviews of #2625 and #2626 on current upstream main 1ef7b8913 (each PR's head with main merged,
# patches/m1ef7b8913_20261012/review_pr<n>.patch): the Fun-CosyVoice3 and sampling unit suites on each
# (t_r2625 card 0, t_r2626 card 1), then the component tests: staged_ids_component (collect_tokens on CUDA
# ids, #2625) and ras_component (the redraw's distribution, seeds and per step cost, #2626).
# usage (in the container): bash queue_rev_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
PATCHES=$R/patches/m1ef7b8913_20261012
MAIN=1ef7b8913cd2a4a3c803f1c75fafad42537356ce
git -C /workspace/sglang-omni fetch -q upstream "$MAIN"
for n in 2625 2626; do
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_r$n "$MAIN"
  git -C /workspace/t_r$n apply -3 $PATCHES/review_pr$n.patch
  git -C /workspace/t_r$n add -A
  git -C /workspace/t_r$n reset -q
  echo "t_r$n on $MAIN with review_pr$n changed=$(git -C /workspace/t_r$n status --short | wc -l)" \
    | tee /workspace/t_r$n/BRANCH_MARKER.txt >> $R/queue_rev.txt
done
mkdir -p $R/rev
(cd /workspace/t_r2625 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_r2625 python3 -m pytest \
  tests/unit_test/fun_cosyvoice3 tests/unit_test/sampling -q -p no:cacheprovider > $R/rev/units_2625.log 2>&1) &
(cd /workspace/t_r2626 && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_r2626 python3 -m pytest \
  tests/unit_test/fun_cosyvoice3 tests/unit_test/sampling -q -p no:cacheprovider > $R/rev/units_2626.log 2>&1) &
wait
echo "units 2625: $(tail -1 $R/rev/units_2625.log)" >> $R/queue_rev.txt
echo "units 2626: $(tail -1 $R/rev/units_2626.log)" >> $R/queue_rev.txt
(cd /workspace/t_r2625 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_r2625 python3 -u $IN/staged_ids_component_c20261012.py) \
  > $R/rev/staged_ids.log 2>&1
echo "staged ids rc $? $(date +%T)" >> $R/queue_rev.txt
(cd /workspace/t_r2626 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_r2626 python3 -u $IN/ras_component_c20261012.py) \
  > $R/rev/ras_component.log 2>&1
echo "ras component rc $? $(date +%T)" >> $R/queue_rev.txt
echo "queue done $(date +%T)" >> $R/queue_rev.txt
