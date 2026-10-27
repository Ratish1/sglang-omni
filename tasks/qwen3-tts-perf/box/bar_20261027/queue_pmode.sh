#!/usr/bin/env bash
# prompt-mode branch: bar arm then main, units with the pre-fix proof, then WER of both arms' recorded audio.
set -u
K=/workspace/kit_9f379d285/tasks/qwen3-tts-perf/box
B=/data/bar_20261027
P=/data/pmode_20261027
mkdir -p $P
until grep -q queue_census_done /data/census2_20261027/queue.log 2> /dev/null; do sleep 20; done
echo "pmode start $(date +%T)" >> $P/log.txt
PORT=34200 LABEL=pmode TREE=/workspace/stack_0fbe8fc01 RATES="1 20" T=/workspace/kit_9f379d285 D=$B bash $K/bar_20261027/queue_baseline.sh > $B/pmode.out 2>&1
PORT=34400 LABEL=main_g TREE=/workspace/main_72b776c9c RATES="1 20" T=/workspace/kit_9f379d285 D=$B bash $K/bar_20261027/queue_baseline.sh > $B/main_g.out 2>&1
(cd /workspace/stack_0fbe8fc01 && timeout 900 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py 2>&1 | grep -E "^FAILED|passed|failed" | tail -n 3) >> $P/log.txt
cp /workspace/stack_0fbe8fc01/tests/unit_test/qwen3_tts/test_pipeline.py /workspace/main_72b776c9c/tests/unit_test/qwen3_tts/test_prefix_pmode.py
(cd /workspace/main_72b776c9c && python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_prefix_pmode.py -k explicit_streaming_text_prompt 2>&1 | grep -E "^E |passed|failed" | head -n 4) >> $P/log.txt
rm /workspace/main_72b776c9c/tests/unit_test/qwen3_tts/test_prefix_pmode.py
for arm in pmode main_g; do
  for rate in 1 20; do
    python3 $K/bar_20261027/bar_wer.py $B/$arm --rate $rate --out $P/${arm}_rps$rate --tree /workspace/main_72b776c9c --card 0 --port 35000 >> $P/log.txt 2>&1
  done
done
echo "queue_pmode_done $(date +%T)" >> $P/log.txt
