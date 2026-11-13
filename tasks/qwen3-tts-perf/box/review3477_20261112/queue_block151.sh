#!/usr/bin/env bash
# Block 151 of 11-13: #3477's whole-request replay order, two H100 on host-85-234-79-221. A shares the tail graph
# rule between first chunks and whole requests, B also continues slot holders ahead of new whole requests, P is the
# PR head before both. The unit suites on A and B and the device check on A, seeded c1 byte identity of P against A
# and of A against B in both modes, then A against B at non-streaming c16 and c64 and open loop at 50 and 70 a second
# in both card orders.
# usage: D=... T=<kit tree> P=<tree> A=<tree> B=<tree> bash queue_block151.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
L=$D/queue.log
mkdir -p $D
echo $$ > $D/queue.pgid
for arm in P A B; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
echo "queue start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
for arm in A B; do
  tree=${!arm}
  (cd $tree && PYTHONPATH=$tree timeout 1500 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline/test_stage.py tests/unit_test/utils/test_audio.py > $D/suite_$arm.log 2>&1)
  echo "suite $arm rc $? $(date +%T): $(grep -E 'passed|failed' $D/suite_$arm.log | tail -1)" >> $L
done
(cd $A && PYTHONPATH=$A QWEN3_TTS_BASE_1_7B_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base timeout 1500 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py -k test_real_base_whole_requests_on_reserved_slots > $D/device.log 2>&1)
echo "device rc $? $(date +%T): $(grep -E 'passed|failed' $D/device.log | tail -1)" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $P --branch $A --root $D/identity_PA --port 24000 > $D/identity_PA.txt 2>&1)
echo "identity P A rc $? $(date +%T): $(grep -E 'byte identical' $D/identity_PA.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $A --branch $B --root $D/identity_AB --port 24600 > $D/identity_AB.txt 2>&1)
echo "identity A B rc $? $(date +%T): $(grep -E 'byte identical' $D/identity_AB.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $K/review3477_20261112 && python3 review_cells_v2.py --main $A --head $B --root $D/cells --phases order1,order2 \
  --card-cpus "0=0-15,64-79;1=16-31,80-95" --client-cpus 32-63,96-127 --port 23000 > $D/cells.txt 2>&1)
echo "cells rc $? $(date +%T)" >> $L
echo "queue done $(date +%T)" >> $L
