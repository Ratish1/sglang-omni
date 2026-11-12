#!/usr/bin/env bash
# Block 147 of 11-12: #3477 after the 11-12 review, two H100 on host-85-234-79-221 (both on NUMA node 0).
# Seeded c1 byte identity of the previous head against the final head, then of main against it, then review_cells_v2's
# mixed, streaming c64, n16 and n64 cells in both orders, the main-against-main band, and both orders without codec graphs.
# usage: D=... T=<kit tree> M=<main tree> P=<previous head tree> F=<final head tree> bash queue_block147.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
L=$D/queue.log
mkdir -p $D
echo $$ > $D/queue.pgid
for arm in M P F; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
echo "queue start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $P --branch $F --root $D/identity_previous --port 24000 > $D/identity_previous.txt 2>&1)
echo "identity previous rc $? $(date +%T): $(grep -E 'byte identical' $D/identity_previous.txt | cut -c1-160 | tr '\n' '#')" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $M --branch $F --root $D/identity_main --port 24400 > $D/identity_main.txt 2>&1)
echo "identity main rc $? $(date +%T): $(grep -E 'byte identical' $D/identity_main.txt | cut -c1-160 | tr '\n' '#')" >> $L
(cd $K/review3477_20261112 && python3 review_cells_v2.py --main $M --head $F --root $D/cells --phases o1m,o2m,bandm,nograph,nograph2 \
  --card-cpus "0=0-15,64-79;1=16-31,80-95" --client-cpus 32-63,96-127 --port 23000 > $D/cells.txt 2>&1)
echo "cells rc $? $(date +%T)" >> $L
echo "queue done $(date +%T)" >> $L
