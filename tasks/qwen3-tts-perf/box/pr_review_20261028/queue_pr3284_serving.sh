#!/usr/bin/env bash
# #3284 redesign, second pass on the H100 (our head retries a refused launch in place): host cost per eager pass first and
# alone (main, ours, three rounds); then on card 0 the stage bits, main against ours PTX without line info, and our
# touched suites, while cards 2 and 3 serve main and ours for the seeded identity (1.7B Base, SeedTTS EN, c1
# non-streaming, seed 1234); then the unseeded non-streaming c16 A/B with WER and SIM, main and ours at once on cards
# 0 and 1, then swapped.
set -u
D=${D:?the results directory}
OURS=${OURS:?our tree}
MAIN=${MAIN:?the main tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/pr_review_20261028
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
L=$D/queue2.log
export ARM_ROOT=$D/arms
mkdir -p $D/arms
echo "queue2 start $(date +%T) ours $(git -C $OURS rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for repeat in 1 2 3; do
  for tree in main ours; do
    if [ $tree = ours ]; then dir=$OURS; else dir=$MAIN; fi
    (cd $dir && PYTHONPATH=$dir python3 $HERE/predictor_fit.py host >> $D/host2_$tree.txt 2>&1)
  done
done
echo "host2 main: $(grep -o '[0-9.]* us' $D/host2_main.txt | tr '\n' ' ')" >> $L
echo "host2 ours: $(grep -o '[0-9.]* us' $D/host2_ours.txt | tr '\n' ' ')" >> $L
(python3 $ARM --label seeded_main --tree $MAIN --card 2 --port 27100 --concurrency 1 --seed 1234 --no-stream >> $D/arms_seeded_main.log 2>&1; echo "arm seeded_main rc $? $(date +%T)" >> $L) &
(python3 $ARM --label seeded_ours --tree $OURS --card 3 --port 27200 --concurrency 1 --seed 1234 --no-stream >> $D/arms_seeded_ours.log 2>&1; echo "arm seeded_ours rc $? $(date +%T)" >> $L) &
(cd $OURS && PYTHONPATH=$OURS python3 $HERE/predictor_fit.py bits > $D/bits2_ours.txt 2>&1)
echo "bits2 ours rc $? not identical $(grep -o 'identical False' $D/bits2_ours.txt | wc -l) $(date +%T)" >> $L
for tree in main ours; do
  if [ $tree = ours ]; then dir=$OURS; else dir=$MAIN; fi
  (cd $dir && PYTHONPATH=$dir python3 $HERE/predictor_fit.py hashes > $D/ptx_$tree.txt 2>&1)
done
echo "ptx without line info differ: $(diff <(grep BLOCK_M $D/ptx_main.txt | sed 's/bytes [0-9a-f]* /bytes /') <(grep BLOCK_M $D/ptx_ours.txt | sed 's/bytes [0-9a-f]* /bytes /') | grep -c '^[<>]') of $(grep -c BLOCK_M $D/ptx_ours.txt) lines $(date +%T)" >> $L
(cd $OURS && python3 -m pytest tests/unit_test/qwen3_omni/test_predictor_kernels.py tests/unit_test/qwen3_omni/test_talker.py \
  tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py -q -p no:cacheprovider > $D/units2_ours.txt 2>&1)
echo "units2 ours: $(tail -n 3 $D/units2_ours.txt | grep -E 'passed|failed|error' | tail -n 1) $(date +%T)" >> $L
wait
python3 $ARM --compare seeded_main seeded_ours >> $L 2>&1
for round in 1 2; do
  if [ $round = 1 ]; then first=main; second=ours; else first=ours; second=main; fi
  for side in 0 1; do
    if [ $side = 0 ]; then tree=$first; else tree=$second; fi
    if [ $tree = ours ]; then dir=$OURS; else dir=$MAIN; fi
    (python3 $ARM --label c16n_${tree}$round --tree $dir --card $side --port $((27300 + 100 * side + 40 * round)) --concurrency 16 \
      --no-stream --score >> $D/arms_c16n_${tree}$round.log 2>&1; echo "arm c16n_${tree}$round card $side rc $? $(date +%T)" >> $L) &
  done
  wait
done
echo "queue2 done $(date +%T)" >> $L
