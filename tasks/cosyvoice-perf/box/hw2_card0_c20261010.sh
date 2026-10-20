#!/usr/bin/env bash
# Step 1 v2 (excitation summed in place) on card 0: tree t_hw2, HiFT GPU tests, timing bench.
R=/data/c20261010h
MAIN=ba456b68f516dd14e841fa150ba708c7801a2b65
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_hw2 $MAIN
git -C /workspace/t_hw2 apply $R/in/hw_ba456b68f_20261010/hift_window_v2.patch
echo "t_hw2 on $MAIN with hift_window_v2 changed=$(git -C /workspace/t_hw2 status --short | wc -l)" | tee /workspace/t_hw2/BRANCH_MARKER.txt >> $R/hw2.txt
cd /workspace/t_hw2
export PYTHONPATH=/workspace/t_hw2
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -m "accelerator and not benchmark" -rs > $R/units/t_hw2.gpu.log 2>&1
echo "gpu rc $? $(date +%T)" >> $R/hw2.txt
CUDA_VISIBLE_DEVICES="" taskset -c 0-15,64-79 python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider -m "not benchmark and not accelerator" -rs > $R/units/t_hw2.cpu.log 2>&1
echo "cpu rc $? $(date +%T)" >> $R/hw2.txt
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 $R/in/hift_window_bench_c20261010.py --out $R/bench/t_hw2.json > $R/bench/t_hw2.log 2>&1
echo "bench rc $? $(date +%T)" >> $R/hw2.txt
CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 $R/in/hift_window_bench_c20261010.py --accuracy --out $R/bench/t_hw2_acc.json > $R/bench/t_hw2_acc.log 2>&1
echo "accuracy rc $? $(date +%T)" >> $R/hw2.txt
echo "done $(date +%T)" >> $R/hw2.txt
