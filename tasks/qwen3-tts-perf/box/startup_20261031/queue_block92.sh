#!/usr/bin/env bash
# Block 92 of 11-02: the same-time c64 pair of the final whole request head against #3418's head, both card orders, and the
# seeded x-vector and reference comparison between the arms on real servers. Waits for block 91.
# usage: D=... T=<kit tree> PR=<tree> WHOLE=<tree> bash queue_block92.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
PR=${PR:?the base tree}
WHOLE=${WHOLE:?the tree under test}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block92
L=$W/queue.log
mkdir -p $W
until grep -q "queue block91 done" $D/block91/queue.log 2> /dev/null; do sleep 5; done
echo "queue block92 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $K/startup_20261031 && python3 whole_final_pair.py --a pr=$PR --b whole=$WHOLE --root $W/pair --references $D/block90/ab/n16_o1/whole/bench --port 26000 > $W/pair.txt 2>&1)
echo "pair rc $? $(date +%T)" >> $L
grep -E 'healthy by|^probe|req/s|Error|Traceback' $W/pair.txt | cut -c1-260 >> $L
echo "queue block92 done $(date +%T)" >> $L
