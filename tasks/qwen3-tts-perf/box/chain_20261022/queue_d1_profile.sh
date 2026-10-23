#!/usr/bin/env bash
# Whether #3102 with the reach kept per emit count (e3768bbc2) leaves any first-audio cost at SeedTTS EN c64, and where
# the vocoder's initial worker spends its time. Two more c64 pairs against main with request events (four in all); then
# two c64 pairs with py-spy on both arms' stage processes (pyspy_sidecar.py): the pre-fix head 7e677616b against the
# fix, and main against the fix. Then the per-request stage gaps pooled per tree with bootstrap intervals
# (stage_gap_compare.py) and the initial worker's split (planner_profile.py).
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
PROF=$T/tasks/qwen3-tts-perf/box/profile_20261022
M=${M:?the main tree}
P1=${P1:?the pre-fix tree}
PC=${PC:?the fix tree}
L=$D/queue.log
echo "queue d1 profile start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) pre $(git -C $P1 rev-parse HEAD:sglang_omni) fix $(git -C $PC rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
pair() {
  local label=$1 tree_a=$2 tree_b=$3 card_a=$4 card_b=$5 port=$6 profile=$7
  if [ $profile = yes ]; then
    python3 $HERE/pyspy_sidecar.py $label --duration 60 > $D/$label.sidecar 2>&1 &
  else
    :
  fi
  python3 $IN/pair.py --label $label --tree $tree_a --b-tree $tree_b --concurrency 64 --a-card $card_a --b-card $card_b \
    --port $port --events > $D/$label.out 2>&1
  local rc=$?
  wait
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(tr '\n' ' ' < $D/$label.sidecar 2>/dev/null)" >> $L
}
pair p1r2_c64_a $M $PC 2 3 25400 no
pair p1r2_c64_b $M $PC 3 2 26600 no
pair p1p_pre_c64 $P1 $PC 2 3 27800 yes
pair p1p_main_c64 $M $PC 3 2 19400 yes
(cd $PROF && python3 stage_gap_compare.py 3a4e73436 e3768bbc2 p1r_c64_a p1r_c64_b p1r2_c64_a p1r2_c64_b p1p_main_c64 \
  > $D/stage_gap_main_fix.txt 2>&1)
(cd $PROF && python3 stage_gap_compare.py 3a4e73436 7e677616b p1e_c64_a p1e_c64_b > $D/stage_gap_main_prefix.txt 2>&1)
(cd $PROF && python3 stage_gap_compare.py 7e677616b e3768bbc2 p1p_pre_c64 > $D/stage_gap_prefix_fix.txt 2>&1)
python3 $HERE/planner_profile.py p1p_pre_c64 p1p_main_c64 > $D/planner_profile_d1.txt 2>&1
echo "queue_d1_profile_done $(date +%T)" >> $L
