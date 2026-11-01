#!/usr/bin/env bash
# Block 59d of 11-01: block 59's main-against-main band served all eight runs and scored card 1's four, but card 0's
# ASR scorer lost port 36400 (inside the ephemeral range) at bind. Card 0's four runs are scored here on a port below
# 28000 (ab_runs.py --score-cards 0), then the band's table. Waits for block 60.
# usage: D=... T=<kit tree> MAIN=<tree> bash queue_block59d.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
MAIN=${MAIN:?the main tree}
K=$T/tasks/qwen3-tts-perf/box
W=$D/block59d
L=$W/queue.log
mkdir -p $W
until grep -q "queue block60 done" $D/block60/queue.log 2> /dev/null; do sleep 20; done
echo "queue block59d start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $K/next_20261031 && python3 ab_runs.py --a maina=$MAIN --b mainb=$MAIN --root $D/block59/band --port 17000 --score-cards 0 > $W/band.txt 2>&1)
echo "band: $(grep -E '^(n16|s16)_o[12]|paired|Error' $W/band.txt | cut -c1-400 | tr '\n' '#')" >> $L
echo "queue block59d done $(date +%T)" >> $L
