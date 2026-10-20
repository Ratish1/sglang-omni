#!/usr/bin/env bash
# V3's startup regression (about 60 s per boot, warm cache included, inside the codec runner phases): per served
# shape, the first eager decode and the precompile on main and on V3, each twice on its own fresh inductor cache so
# the first process is a cold boot and the second a warm one. Runs alone on card 0 between queue_v3.sh and
# queue_v5.sh, since a compile's CPU load would perturb a pair.
set -u
D=/data/v3_20261020
T=${T:-/workspace/t_marks}
S=$T/tasks/omni_step_profiling_20260918/scripts
M=/workspace/m_bcd0d8c68
V3=/workspace/v3_6f6039ba6
L=$D/queue.log
until grep -q queue_v3_done $L; do sleep 30; done
mkdir -p $D/precompile
for TREE in $M $V3; do
  NAME=$(basename $TREE)
  rm -rf $D/precompile/inductor_$NAME
  for PASS in cold warm; do
    (cd $TREE && TORCHINDUCTOR_CACHE_DIR=$D/precompile/inductor_$NAME PYTHONPATH=$TREE timeout 1800 \
      python3 $S/vocoder_precompile_bench.py > $D/precompile/${NAME}_$PASS.txt 2>&1)
    echo "precompile $NAME $PASS rc $? $(grep -h '^total' $D/precompile/${NAME}_$PASS.txt) $(date +%T)" >> $L
  done
done
echo "queue_precompile_done $(date +%T)" >> $L
