#!/usr/bin/env bash
# After queue_d1_profile.sh, whose sidecar attached only as the servers shut down (the stage process's events file is
# first written at the end of the benchmark): the two c64 profile pairs again with pyspy_sidecar.py attaching once both
# servers are healthy, the pre-fix head 7e677616b against the fix and main against the fix, then planner_profile.py.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
M=${M:?the main tree}
P1=${P1:?the pre-fix tree}
PC=${PC:?the fix tree}
L=$D/queue.log
until grep -q queue_d1_profile_done $L; do sleep 20; done
echo "queue d1 pyspy start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
profile_pair() {
  local label=$1 tree_a=$2 tree_b=$3 card_a=$4 card_b=$5 port=$6
  python3 $HERE/pyspy_sidecar.py $label --port $port --duration 60 > $D/$label.sidecar 2>&1 &
  python3 $IN/pair.py --label $label --tree $tree_a --b-tree $tree_b --concurrency 64 --a-card $card_a --b-card $card_b \
    --port $port --events > $D/$label.out 2>&1
  local rc=$?
  wait
  local failed_a=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/a/serve.log)
  local failed_b=$(grep -ciE 'capture_failed|capture disabled|Unsupported:' $PAIR_ROOT/$label/b/serve.log)
  echo "$label rc $rc $(date +%T) capture or compile failures a $failed_a b $failed_b $(tr '\n' ' ' < $D/$label.sidecar)" >> $L
}
profile_pair p1q_pre_c64 $P1 $PC 2 3 13600
profile_pair p1q_main_c64 $M $PC 3 2 17000
python3 $HERE/planner_profile.py p1q_pre_c64 p1q_main_c64 > $D/planner_profile_d1.txt 2>&1
echo "queue_d1_pyspy_done $(date +%T)" >> $L
