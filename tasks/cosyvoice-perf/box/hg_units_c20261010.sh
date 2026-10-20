#!/usr/bin/env bash
# Step 2 (HiFT from CUDA graphs) on step 1, tree t_hg on main ba456b68f: the Fun-CosyVoice3 unit
# suites, accelerator lane on card 0 and CPU lane.
R=/data/c20261010h
MAIN=ba456b68f516dd14e841fa150ba708c7801a2b65
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_hg $MAIN
git -C /workspace/t_hg apply $R/in/hw_ba456b68f_20261010/hift_window_graphs_v1.patch
echo "t_hg on $MAIN with hift_window_graphs_v1 changed=$(git -C /workspace/t_hg status --short | wc -l)" | tee /workspace/t_hg/BRANCH_MARKER.txt >> $R/hg.txt
cd /workspace/t_hg
export PYTHONPATH=/workspace/t_hg
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -m "accelerator and not benchmark" -rs > $R/units/t_hg.gpu.log 2>&1
echo "gpu rc $? $(date +%T)" >> $R/hg.txt
CUDA_VISIBLE_DEVICES="" taskset -c 0-15,64-79 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -m "not benchmark and not accelerator" -rs > $R/units/t_hg.cpu.log 2>&1
echo "cpu rc $? $(date +%T)" >> $R/hg.txt
echo "done $(date +%T)" >> $R/hg.txt
