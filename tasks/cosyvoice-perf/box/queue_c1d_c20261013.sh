#!/usr/bin/env bash
# C1 with D served, before any repo edit: t_c1 (main 204ba29a5 unchanged) with c1_patch first on PYTHONPATH and
# HIFT_SPECTRA=1 (patch_sine_harmonics and patch_hift_spectra, DESIGN_NEXT_20261012.md 13 and 14).
# 1. the component check of C1 with D against main on card 0 (outputs compared, step times, blocking census),
# 2. the profile of main (t_m, card 0) and C1 with D (t_c1, card 1) at the same time,
# 3. same time pairs in both card orders, seeded c1 and c16 with WER and SIM.
# usage (in the container): bash queue_c1d_c20261013.sh
set -u
R=/data/c20261013
IN=$R/in
OUT=$R/c1d
PATCH=$IN/c1_patch
mkdir -p $OUT
cd /workspace/t_m
CUDA_VISIBLE_DEVICES=0 C1_PATCH=1 HIFT_SPECTRA=1 TORCHINDUCTOR_CACHE_DIR=$R/inductor_t_m PYTHONPATH=/workspace/t_m \
  taskset -c 0-15,64-79 python3 $IN/c1_sine_component_c20261013.py $OUT/c1d_card0.pt > $OUT/component.log 2>&1
echo "component rc $? $(date +%T)" >> $OUT/queue.txt
python3 $IN/c1_sine_component_c20261013.py --compare $R/c1comp/main_card0.pt $OUT/c1d_card0.pt > $OUT/compare.txt 2>&1

echo "profile $(date +%T)" >> $OUT/queue.txt
PEER=$OUT/prof/t_c1 bash $IN/cosy_profile_run_c20261013.sh t_m 0 0-15,64-79 45000 $OUT/prof/t_m &
PEER=$OUT/prof/t_m EXTRA_PATH=$PATCH HIFT_SPECTRA=1 bash $IN/cosy_profile_run_c20261013.sh t_c1 1 16-31,80-95 45200 \
  $OUT/prof/t_c1 &
wait
echo "profile c1_patch $(grep -c 'c1_patch applied' $OUT/prof/t_c1/serve.log) spectra" \
  "$(grep -c 'hift_spectra applied' $OUT/prof/t_c1/serve.log) main $(grep -c 'c1_patch' $OUT/prof/t_m/serve.log)" \
  >> $OUT/queue.txt

export ROOT=$R
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
export SCORE_CELLS="c16"
export CELLS="c1:1:64:1234:1 c16:16:::1"
echo "pa a $(date +%T) card0=t_m card1=t_c1" >> $OUT/queue.txt
env $PAIR EXTRA_ENV_1="PYTHONPATH=/workspace/t_c1:$PATCH HIFT_SPECTRA=1" bash $IN/ab_pair_sync.sh $OUT/a t_m t_c1 45400
echo "pa b $(date +%T) card0=t_c1 card1=t_m" >> $OUT/queue.txt
env $PAIR EXTRA_ENV_0="PYTHONPATH=/workspace/t_c1:$PATCH HIFT_SPECTRA=1" bash $IN/ab_pair_sync.sh $OUT/b t_c1 t_m 45800
for pair in a b; do
  echo "pa $pair spectra applied: $(grep -c 'hift_spectra applied' $OUT/$pair/card*_t_c1/serve.log)," \
    "in main: $(grep -c 'c1_patch' $OUT/$pair/card*_t_m/serve.log)" >> $OUT/queue.txt
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
