#!/usr/bin/env bash
# qwen-tts's embedding against the server's for the same references, 0.6B on card 2 and 1.7B on card 3, once the #3284
# seeded identity has left both cards (both arms' lines in AFTER_LOG).
set -u
D=${D:?the pr2533 results directory}
TREE=${TREE:?a tree with the compatibility patches}
T=${T:?the kit tree}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
HERE=$T/tasks/qwen3-tts-perf/box/pr_review_20261028
L=$D/qwen_tts_compare.log
until [ "$(grep -c 'arm seeded_' $AFTER_LOG 2> /dev/null)" -ge 2 ]; do
  sleep 20
done
echo "qwen-tts compare start $(date +%T) tree $(git -C $TREE rev-parse HEAD) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $TREE && PYTHONPATH=$TREE python3 $HERE/qwen_tts_embedding_compare.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --dump-dir $D/dump06 --device cuda:2 > $D/qwen_tts_compare06.txt 2>&1; echo "0.6B rc $?: $(tail -n 1 $D/qwen_tts_compare06.txt) $(date +%T)" >> $L) &
(cd $TREE && PYTHONPATH=$TREE python3 $HERE/qwen_tts_embedding_compare.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --dump-dir $D/dump17 --device cuda:3 > $D/qwen_tts_compare17.txt 2>&1; echo "1.7B rc $?: $(tail -n 1 $D/qwen_tts_compare17.txt) $(date +%T)" >> $L) &
wait
echo "qwen-tts compare done $(date +%T)" >> $L
