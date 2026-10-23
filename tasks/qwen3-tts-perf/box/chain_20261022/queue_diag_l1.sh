#!/usr/bin/env bash
# After queue_prs4.sh: D8 alone on main holds a long reference's first audio near 1.06 s at L1 c16 where main's falls
# to about 0.5 s after the opening wave (p8_L1_c16_a and b). Request events on both arms, main against the whole of D8,
# then main against D8's first commit alone (the speaker path; the second commit is the reference encode's rounding),
# L1 c16, unseeded, no scoring; each arm's first-audio chain from its events.
set -u
export PAIR_ROOT=/data/prs_20261022/pairs
D=/data/prs_20261022
T=${T:-/workspace/t_marks}
IN=$T/tasks/qwen3-tts-perf/box/v3_20261020
CHAIN=$T/tasks/qwen3-tts-perf/box/profile_20261022/first_audio_chain.py
LONG=/data/prof_20261022/formal/meta_w2/long
M=${M:?the main tree}
P8=${P8:?the D8 tree}
P8S=${P8S:?the tree at the first D8 commit}
L=$D/queue.log
until grep -q queue_prs4_done $L; do sleep 20; done
echo "queue diag l1 start $(date +%T) main $(git -C $M rev-parse HEAD:sglang_omni) d8 $(git -C $P8 rev-parse HEAD:sglang_omni) d8 speaker $(git -C $P8S rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for run in "diag_L1_d8 $P8 2 3 14800" "diag_L1_d8speaker $P8S 3 2 17000"; do
  set -- $run
  python3 $IN/pair.py --label $1 --tree $M --b-tree $2 --concurrency 16 --a-card $3 --b-card $4 --port $5 \
    --meta $LONG/L1_c16_run.lst --skip-score --events > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
  python3 $CHAIN $PAIR_ROOT/$1/a/events $PAIR_ROOT/$1/b/events > $D/chain_$1.txt 2>&1
done
echo "queue_diag_l1_done $(date +%T)" >> $L
