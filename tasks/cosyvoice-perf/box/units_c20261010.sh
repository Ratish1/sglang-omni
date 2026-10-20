#!/usr/bin/env bash
# The Fun-CosyVoice3 unit suites (accelerator lane, then CPU lane) of every tree of
# setup_c20261010.sh, on cards 0 and 3 in two lanes.
# usage (in the container): bash units_c20261010.sh
R=/data/c20261010
mkdir -p $R/units

units() {
  local tree=$1 card=$2 cpus=$3
  (
    cd /workspace/$tree
    export PYTHONPATH=/workspace/$tree CUDA_VISIBLE_DEVICES=$card
    python3 -c "import sglang_omni; print(sglang_omni.__file__)" > $R/units/$tree.import.txt
    taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q -p no:cacheprovider \
      -m "accelerator and not benchmark" -rs > $R/units/$tree.gpu.log 2>&1
    echo "gpu rc $? $(date +%T)" >> $R/units/$tree.done
    CUDA_VISIBLE_DEVICES="" taskset -c "$cpus" python3 -m pytest tests/unit_test/fun_cosyvoice3 -q \
      -p no:cacheprovider -m "not benchmark and not accelerator" -rs > $R/units/$tree.cpu.log 2>&1
    echo "cpu rc $? $(date +%T)" >> $R/units/$tree.done
  )
}

echo "units start $(date +%T)" >> $R/units/progress.txt
(for tree in t_stack t_p1 t_i3 sglang-omni; do units $tree 0 0-15,64-79; done) &
(for tree in t_stack5 t_p2 t_i4 t_i5; do units $tree 3 16-31,80-95; done) &
wait
echo "units done $(date +%T)" >> $R/units/progress.txt
