#!/usr/bin/env bash
# Block 48 of 10-31: PR B's one-program test at widths 4 and 8 (block 45 ran it at 2 and 8, which the test fixture's
# dilation-9 conv guard splits) on HEAD and on BASE (PR A, where it must fail), card 1. Waits for block 47 first.
# usage: D=... BASE=<tree> HEAD=<tree> bash queue_block48.sh
set -u
D=${D:?the results directory}
BASE=${BASE:?the PR A tree}
HEAD=${HEAD:?the PR B tree}
TEST=tests/unit_test/qwen3_tts/test_incremental_codec.py::test_compiled_decode_serves_every_batch_and_width_above_one_with_one_program
W=$D/block48
L=$W/queue.log
mkdir -p $W
until grep -q "queue block47 done" $D/block47/queue.log 2> /dev/null; do sleep 20; done
for tree in $BASE $HEAD; do
  echo "tree $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
(cd $HEAD && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$HEAD python3 -m pytest -q -p no:cacheprovider $TEST > $W/test_head.log 2>&1)
echo "new test on HEAD exit $? $(grep -E 'passed|failed' $W/test_head.log | tail -n 1)" >> $L
cp $HEAD/tests/unit_test/qwen3_tts/test_incremental_codec.py $BASE/tests/unit_test/qwen3_tts/test_incremental_codec.py
(cd $BASE && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$BASE python3 -m pytest -q -p no:cacheprovider $TEST > $W/test_base.log 2>&1)
echo "new test on BASE exit $? $(grep -E 'passed|failed' $W/test_base.log | tail -n 1) $(grep -E '^E ' $W/test_base.log | head -n 2 | tr '\n' ' ')" >> $L
git -C $BASE checkout -q -- tests
echo "queue block48 done $(date +%T)" >> $L
