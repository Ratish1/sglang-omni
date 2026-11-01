#!/usr/bin/env bash
# Block 62 of 11-01: #3403's review (tasks/pr3403_review_20261101, P4) asks for a case that runs the codec's opaque ops
# through a real torch.compile and CUDA graph replay. The candidate head adds it; the qwen3_tts suite runs on it, then
# the new case alone on a scratch tree whose sum op declares no writes, which the case must reject. Waits for block 61b.
# usage: D=... T=<kit tree> CANDIDATE=<tree> BAD=<tree> bash queue_block62.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
CANDIDATE=${CANDIDATE:?the #3403 candidate tree}
BAD=${BAD:?the broken-schema tree}
W=$D/block62
L=$W/queue.log
mkdir -p $W
until grep -q "queue block61b done" $D/block61b/queue.log 2> /dev/null; do sleep 20; done
for arm in CANDIDATE BAD; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block62 start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $CANDIDATE && CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/suite.log 2>&1)
echo "suite: $(grep -E '[0-9]+ (passed|failed)' $W/suite.log | tail -n 1)" >> $L
(cd $CANDIDATE && CUDA_VISIBLE_DEVICES=0 timeout 900 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_incremental_codec.py -k compiled_slot_ops > $W/new_case.log 2>&1)
echo "new case on the candidate: $(grep -E '[0-9]+ (passed|failed)' $W/new_case.log | tail -n 1)" >> $L
(cd $BAD && CUDA_VISIBLE_DEVICES=0 timeout 900 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_incremental_codec.py -k compiled_slot_ops > $W/bad_schema.log 2>&1)
echo "new case with the sum op declaring no writes: $(grep -E '[0-9]+ (passed|failed)' $W/bad_schema.log | tail -n 1) $(grep -m1 -E '^E ' $W/bad_schema.log | cut -c1-160)" >> $L
echo "queue block62 done $(date +%T)" >> $L
