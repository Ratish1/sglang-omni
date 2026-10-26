#!/usr/bin/env bash
# The inline branch's second commit 996bb56c6 (the reflect-padded local renamed padded_waveform) on the tested
# b4b7331c2: qwen3_tts units, then the speaker bench on SeedTTS EN, the L3 and the 200 s references, each set's
# embeddings against the PR's from queue_inline.sh (max |delta| 0 expected). One process on card 0.
set -u
D=/data/inline_20261026
T=${T:?the kit tree}
R=${R:?the renamed tree}
BENCH=$T/tasks/qwen3-tts-perf/box/chain_20261022/speaker_embed_bench.py
L=$D/queue.log
echo "queue rename start $(date +%T) renamed $(git -C $R rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $R diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
(cd $R && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$R timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q -p no:cacheprovider \
  -rfEs > $D/units/renamed.log 2>&1)
echo "units renamed rc $? $(grep -E 'passed|failed' $D/units/renamed.log | tail -n 1) $(date +%T)" >> $L
(cd $R && PYTHONPATH=$R python3 $BENCH $D/speaker_seedtts_renamed.pt --card 0 > $D/speaker_seedtts_renamed.txt 2>&1)
(cd $R && PYTHONPATH=$R python3 $BENCH $D/speaker_L3_renamed.pt --card 0 --repeats 5 --meta $D/long/L3_c16_run.lst \
  > $D/speaker_L3_renamed.txt 2>&1)
(cd $R && PYTHONPATH=$R python3 $BENCH $D/speaker_ref200_renamed.pt --card 0 --repeats 5 --meta $D/ref200/ref200s.lst \
  > $D/speaker_ref200_renamed.txt 2>&1)
for set_name in seedtts L3 ref200; do
  echo "speaker $set_name renamed: $(grep -E 'embed wall|Error' $D/speaker_${set_name}_renamed.txt | tail -n 1)" >> $L
  (cd $R && PYTHONPATH=$R python3 $BENCH --compare $D/speaker_${set_name}_pr.pt $D/speaker_${set_name}_renamed.pt \
    > $D/speaker_compare_${set_name}_pr_renamed.txt 2>&1)
  echo "speaker $set_name pr against renamed: $(tail -n 1 $D/speaker_compare_${set_name}_pr_renamed.txt)" >> $L
done
echo "queue_rename_done $(date +%T)" >> $L
