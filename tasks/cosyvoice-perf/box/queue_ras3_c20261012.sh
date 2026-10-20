#!/usr/bin/env bash
# Our final #2626 branch (ab694915e: the redraw in collect_tokens without the RL, Ascend, logprob and TP handling
# CosyVoice3 never reaches, fixed_pr2626_v4.patch) on main 1ef7b8913 as t_f2626d, after queue_ras2's pairs: the
# rows only redraw component test and the Fun-CosyVoice3, sampling and model_runner suites.
# usage (in the container): bash queue_ras3_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
MAIN=1ef7b8913cd2a4a3c803f1c75fafad42537356ce
until grep -q "queue done" $R/queue_ras2.txt 2>/dev/null; do sleep 30; done
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_f2626d "$MAIN"
git -C /workspace/t_f2626d apply -3 $R/patches/m1ef7b8913_20261012/fixed_pr2626_v4.patch
git -C /workspace/t_f2626d add -A
git -C /workspace/t_f2626d reset -q
echo "t_f2626d on $MAIN with fixed_pr2626_v4 changed=$(git -C /workspace/t_f2626d status --short | wc -l)" \
  | tee /workspace/t_f2626d/BRANCH_MARKER.txt >> $R/queue_ras3.txt
mkdir -p $R/ras3
(cd /workspace/t_f2626d && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_f2626d python3 -u $IN/ras_rows_component_c20261012.py) \
  > $R/ras3/component.log 2>&1
echo "component rc $? $(date +%T)" >> $R/queue_ras3.txt
(cd /workspace/t_f2626d && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_f2626d python3 -m pytest \
  tests/unit_test/fun_cosyvoice3 tests/unit_test/sampling tests/unit_test/model_runner -q -p no:cacheprovider -rfE \
  > $R/ras3/units.log 2>&1)
echo "units: $(grep -E '[0-9]+ (passed|failed)' $R/ras3/units.log | tail -1) $(date +%T)" >> $R/queue_ras3.txt
echo "queue done $(date +%T)" >> $R/queue_ras3.txt
