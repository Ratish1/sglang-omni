#!/usr/bin/env bash
# The whole unit suite on main (t_m1) and our #2625 and #2626 branches (t_f2625, t_f2626), after queue_fix's pairs:
# the first try stopped at collection on MiniCPM-o, whose tests import onnx (its extras), the same on main. onnx
# installed, every suite run with collection errors allowed, then the failures of each branch against main's.
# usage (in the container): bash queue_units_c20261012.sh
set -u
R=/data/c20261012
until grep -q "queue done" $R/queue_fix.txt 2>/dev/null; do sleep 30; done
uv pip install -q --python "$(command -v python3)" onnx >> $R/queue_units.txt 2>&1
echo "onnx $(python3 -c 'import onnx; print(onnx.__version__)') $(date +%T)" >> $R/queue_units.txt
units() {
  local tree=$1 card=$2
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=$card PYTHONPATH=/workspace/$tree timeout 5400 python3 -m pytest \
    tests/unit_test -q -p no:cacheprovider -rfE --continue-on-collection-errors > $R/fix/units2_$tree.log 2>&1)
  echo "units $tree: $(grep -E '[0-9]+ (passed|failed)' $R/fix/units2_$tree.log | tail -1) $(date +%T)" >> $R/queue_units.txt
  grep -E "^(FAILED|ERROR) " $R/fix/units2_$tree.log | sed 's/ - .*//' | sort > $R/fix/failures_$tree.txt
}
units t_m1 0 &
units t_f2626 1 &
wait
units t_f2625 0
for tree in t_f2625 t_f2626; do
  echo "$tree failing where main passes: $(comm -13 $R/fix/failures_t_m1.txt $R/fix/failures_$tree.txt | wc -l)" >> $R/queue_units.txt
  comm -13 $R/fix/failures_t_m1.txt $R/fix/failures_$tree.txt >> $R/queue_units.txt
done
echo "queue done $(date +%T)" >> $R/queue_units.txt
