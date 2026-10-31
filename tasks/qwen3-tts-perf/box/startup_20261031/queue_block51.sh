#!/usr/bin/env bash
# Block 51 of 10-31: C's Qwen3-TTS suite again on its corrected head (the builder keeps its placeholder checkpoint_dir),
# card 1. Waits for block 50 first.
# usage: D=... LAUNCHER=<tree> bash queue_block51.sh
set -u
D=${D:?the results directory}
LAUNCHER=${LAUNCHER:?the launcher tree}
W=$D/block51
L=$W/queue.log
mkdir -p $W
until grep -q "queue block50 done" $D/block50/queue.log 2> /dev/null; do sleep 20; done
echo "tree $LAUNCHER $(git -C $LAUNCHER rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $LAUNCHER diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
(cd $LAUNCHER && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$LAUNCHER timeout 3600 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts > $W/launcher_suite.log 2>&1)
echo "launcher_suite exit $? $(grep -E '(passed|failed|error)' $W/launcher_suite.log | tail -n 1) $(date +%T)" >> $L
echo "queue block51 done $(date +%T)" >> $L
