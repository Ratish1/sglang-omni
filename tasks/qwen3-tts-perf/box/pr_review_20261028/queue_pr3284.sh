#!/usr/bin/env bash
# #3284 redesign on the H100: compiled bytes per target, stage bit identity, main against ours kernel hashes, the
# touched suites on both trees, the new case called against main, and the step-down's host cost per pass.
set -u
D=${D:?the results directory}
OURS=${OURS:?our tree}
MAIN=${MAIN:?the main tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/pr_review_20261028
L=$D/queue.log
mkdir -p $D
echo "queue pr3284 start $(date +%T) ours $(git -C $OURS rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for arch in 80 86 89 90 100 120 121; do
  (cd $OURS && PYTHONPATH=$OURS python3 $HERE/predictor_fit.py bytes --arch $arch > $D/bytes_sm$arch.txt 2>&1; echo "bytes sm$arch rc $? lines $(grep -c bytes $D/bytes_sm$arch.txt) $(date +%T)" >> $L) &
done
wait
for tree in ours main; do
  if [ $tree = ours ]; then dir=$OURS; else dir=$MAIN; fi
  (cd $dir && PYTHONPATH=$dir python3 $HERE/predictor_fit.py bits > $D/bits_$tree.txt 2>&1)
  echo "bits $tree rc $? $(date +%T)" >> $L
  (cd $dir && PYTHONPATH=$dir python3 $HERE/predictor_fit.py hashes > $D/hashes_$tree.txt 2>&1)
  echo "hashes $tree rc $? $(date +%T)" >> $L
done
echo "hashes differ: $(diff <(grep BLOCK_M $D/hashes_main.txt) <(grep BLOCK_M $D/hashes_ours.txt) | grep -c '^[<>]') lines" >> $L
(cd $MAIN && PYTHONPATH=$MAIN python3 $HERE/prefix_stage_test.py $OURS/tests/unit_test/qwen3_omni/test_predictor_kernels.py > $D/prefix_main.txt 2>&1)
echo "prefix main: $(tail -n 1 $D/prefix_main.txt | cut -c1-200) $(date +%T)" >> $L
for tree in ours main; do
  if [ $tree = ours ]; then dir=$OURS; else dir=$MAIN; fi
  (cd $dir && python3 -m pytest tests/unit_test/qwen3_omni/test_predictor_kernels.py tests/unit_test/qwen3_omni/test_talker.py \
    tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py -q -p no:cacheprovider > $D/units_$tree.txt 2>&1)
  echo "units $tree: $(tail -n 3 $D/units_$tree.txt | grep -E 'passed|failed|error' | tail -n 1) $(date +%T)" >> $L
done
for repeat in 1 2 3; do
  for tree in main ours; do
    if [ $tree = ours ]; then dir=$OURS; else dir=$MAIN; fi
    (cd $dir && PYTHONPATH=$dir python3 $HERE/predictor_fit.py host >> $D/host_$tree.txt 2>&1)
  done
done
echo "host main: $(grep -o '[0-9.]* us' $D/host_main.txt | tr '\n' ' ')" >> $L
echo "host ours: $(grep -o '[0-9.]* us' $D/host_ours.txt | tr '\n' ' ')" >> $L
echo "queue pr3284 done $(date +%T)" >> $L
