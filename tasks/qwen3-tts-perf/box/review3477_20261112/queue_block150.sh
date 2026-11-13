#!/usr/bin/env bash
# Block 150 of 11-13: #3477's result built with the audio payload helper from a float32 buffer (B) against the PR
# head that carried a bytearray (A), two H100 on host-85-234-79-221. The unit suite and the device check on B, seeded
# c1 byte identity of A against B in both modes, then non-streaming c16, c64 twice and c1 in both card orders.
# usage: D=... T=<kit tree> A=<tree> B=<tree> bash queue_block150.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
L=$D/queue.log
mkdir -p $D
echo $$ > $D/queue.pgid
for arm in A B; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')" >> $L
done
echo "queue start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $B && PYTHONPATH=$B timeout 1500 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline/test_stage.py tests/unit_test/utils/test_audio.py > $D/suite.log 2>&1)
echo "suite rc $? $(date +%T): $(grep -E 'passed|failed' $D/suite.log | tail -1)" >> $L
(cd $B && PYTHONPATH=$B QWEN3_TTS_BASE_1_7B_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base timeout 1500 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py -k test_real_base_whole_requests_on_reserved_slots > $D/device.log 2>&1)
echo "device rc $? $(date +%T): $(grep -E 'passed|failed' $D/device.log | tail -1)" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $A --branch $B --root $D/identity --port 24000 > $D/identity.txt 2>&1)
echo "identity rc $? $(date +%T): $(grep -E 'byte identical' $D/identity.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $K/review3477_20261112 && python3 review_cells_v2.py --main $A --head $B --root $D/cells --phases w1,w2 \
  --card-cpus "0=0-15,64-79;1=16-31,80-95" --client-cpus 32-63,96-127 --port 23000 > $D/cells.txt 2>&1)
echo "cells rc $? $(date +%T)" >> $L
echo "queue done $(date +%T)" >> $L
