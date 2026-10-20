#!/usr/bin/env bash
# The seeded redraw compile check (seeded_redraw_compile_c20261012.py) on #2626 as reviewed (t_r2626, float32 redraw)
# and on our branch (t_f2626, float64 redraw), card 0, after queue_fix's pairs, before queue_units' suites start.
# usage (in the container): bash queue_compile_c20261012.sh
set -u
R=/data/c20261012
until grep -q "queue done" $R/queue_fix.txt 2>/dev/null; do sleep 15; done
for tree in t_r2626 t_f2626; do
  echo "== $tree" >> $R/fix/seeded_compile.log
  (cd /workspace/$tree && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/$tree TORCHINDUCTOR_CACHE_DIR=$R/inductor_compile_$tree \
    python3 -u /data/c20261012/in/seeded_redraw_compile_c20261012.py) 2>&1 | grep -v "torchada\|Warn" >> $R/fix/seeded_compile.log
done
echo "queue done $(date +%T)" >> $R/fix/seeded_compile.log
