#!/usr/bin/env bash
# C1's component check (c1_sine_component_c20261013.py) on main 204ba29a5 (t_m): unpatched and patched at the same
# time on the two cards, then swapped, then the outputs compared bit for bit across arms and cards.
# usage (in the container): bash queue_c1_component_c20261013.sh
R=/data/c20261013
IN=$R/in
OUT=$R/c1comp
mkdir -p $OUT
cd /workspace/t_m
arm() {
  local name=$1 card=$2 cpus=$3 patch=$4
  CUDA_VISIBLE_DEVICES=$card C1_PATCH=$patch TORCHINDUCTOR_CACHE_DIR=$R/inductor_t_m PYTHONPATH=/workspace/t_m \
    taskset -c $cpus python3 $IN/c1_sine_component_c20261013.py $OUT/$name.pt > $OUT/$name.log 2>&1
  echo "$name rc $? $(date +%T)" >> $OUT/queue.txt
}
echo "start $(date +%T)" >> $OUT/queue.txt
arm main_card0 0 0-15,64-79 0 &
arm c1_card1 1 16-31,80-95 1 &
wait
arm c1_card0 0 0-15,64-79 1 &
arm main_card1 1 16-31,80-95 0 &
wait
python3 $IN/c1_sine_component_c20261013.py --compare $OUT/main_card0.pt $OUT/c1_card1.pt $OUT/c1_card0.pt \
  $OUT/main_card1.pt > $OUT/compare.txt 2>&1
echo "queue done $(date +%T)" >> $OUT/queue.txt
