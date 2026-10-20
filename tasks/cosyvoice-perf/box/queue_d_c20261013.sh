#!/usr/bin/env bash
# Two measurements, no repo edit, ports below the ephemeral range (32768 to 60999 on the box):
# 1. the profile of main (t_m, card 0) and C1 with D (t_c1, c1_patch with HIFT_SPECTRA=1, card 1) at the same
#    time, the run of 09:08 having lost its main arm to a port another process held,
# 2. D alone (t_d, main 204ba29a5 unchanged, c1_patch with HIFT_SINE=0 HIFT_SPECTRA=1) against main, same time
#    pairs in both card orders, seeded c1 and c16 with WER and SIM.
# usage (in the container): bash queue_d_c20261013.sh
set -u
R=/data/c20261013
IN=$R/in
MAIN=204ba29a5
OUT=$R/d
PATCH=$IN/c1_patch
mkdir -p $OUT
[ -d /workspace/t_d ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_d "$MAIN"
echo "t_d on $MAIN with nothing changed=$(git -C /workspace/t_d status --short | wc -l), D alone by c1_patch" \
  | tee /workspace/t_d/BRANCH_MARKER.txt >> $OUT/queue.txt
[ -d $R/inductor_t_d ] || cp -r $R/inductor_t_m $R/inductor_t_d

echo "profile $(date +%T)" >> $OUT/queue.txt
PEER=$OUT/prof/t_c1 bash $IN/cosy_profile_run_c20261013.sh t_m 0 0-15,64-79 21000 $OUT/prof/t_m &
PEER=$OUT/prof/t_m EXTRA_PATH=$PATCH HIFT_SPECTRA=1 bash $IN/cosy_profile_run_c20261013.sh t_c1 1 16-31,80-95 21600 \
  $OUT/prof/t_c1 &
wait
echo "profile c1_patch $(grep -c 'c1_patch applied' $OUT/prof/t_c1/serve.log) spectra" \
  "$(grep -c 'hift_spectra applied' $OUT/prof/t_c1/serve.log) main $(grep -c 'c1_patch' $OUT/prof/t_m/serve.log)" \
  >> $OUT/queue.txt

export ROOT=$R
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
export SCORE_CELLS="c16"
export CELLS="c1:1:64:1234:1 c16:16:::1"
D_ENV="PYTHONPATH=/workspace/t_d:$PATCH HIFT_SINE=0 HIFT_SPECTRA=1"
echo "pa a $(date +%T) card0=t_m card1=t_d" >> $OUT/queue.txt
env $PAIR EXTRA_ENV_1="$D_ENV" bash $IN/ab_pair_sync.sh $OUT/a t_m t_d 22000
echo "pa b $(date +%T) card0=t_d card1=t_m" >> $OUT/queue.txt
env $PAIR EXTRA_ENV_0="$D_ENV" bash $IN/ab_pair_sync.sh $OUT/b t_d t_m 23000
for pair in a b; do
  echo "pa $pair spectra applied: $(grep -c 'hift_spectra applied' $OUT/$pair/card*_t_d/serve.log)," \
    "sine: $(grep -c 'c1_patch applied' $OUT/$pair/card*_t_d/serve.log), in main: $(grep -c 'c1_patch' \
    $OUT/$pair/card*_t_m/serve.log)" >> $OUT/queue.txt
done
echo "queue done $(date +%T)" >> $OUT/queue.txt
