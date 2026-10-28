#!/usr/bin/env bash
# #2533 at the head whose code equals the tested ef13308d4 plus upstream main's later commits, after AFTER_LOG holds
# AFTER_LINE: the suites and a 0.6B Base server on card 1 for the contract, then the seeded identity of plain reference
# requests, main against ours (1.7B Base, c1 non-streaming, cards 1 and 2).
set -u
D=${D:?the results directory}
OURS=${OURS:?our tree}
MAIN=${MAIN:?the main tree}
HEAD_SHA=${HEAD_SHA:?the head to test}
T=${T:?the kit tree}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
HERE=$T/tasks/qwen3-tts-perf/box/pr_review_20261028
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
L=$D/last.log
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
git -C $OURS checkout -q $HEAD_SHA
echo "queue pr2533 last start $(date +%T) ours $(git -C $OURS rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $OURS && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$OURS setsid bash -c "echo \$\$ > $D/last_serve.pid; exec python3 -m sglang_omni.cli serve \
  --host 127.0.0.1 --port 4970 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --tts_engine.gpu 1 --vocoder.gpu 1" > $D/last_serve.log 2>&1 < /dev/null &)
(cd $OURS && python3 -m pytest tests/unit_test/serve tests/unit_test/qwen3_tts tests/unit_test/scheduling/test_pipeline_state.py \
  -q -p no:cacheprovider > $D/last_units.txt 2>&1)
echo "units ours: $(tail -n 3 $D/last_units.txt | grep -E 'passed|failed|error' | tail -n 1); failing: $(grep -E '^FAILED' $D/last_units.txt | sed 's/ - .*//;s/.*:://' | tr '\n' ' ') $(date +%T)" >> $L
waited=0
until curl -sf http://127.0.0.1:4970/health > /dev/null || [ $waited -ge 1800 ]; do sleep 5; waited=$((waited + 5)); done
echo "base06 healthy after $waited s $(date +%T)" >> $L
(cd $OURS && PYTHONPATH=$OURS python3 $HERE/speaker_embedding_e2e.py contract --base-url http://127.0.0.1:4970 \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --embedding $(ls $D/dump06/*.pt | head -n 2 | tail -n 1) \
  --reference $(ls $D/prompt_wavs/*.wav | head -n 1) > $D/last_contract06.txt 2>&1)
echo "contract 06 rc $?, tracebacks $(grep -c Traceback $D/last_serve.log) $(date +%T)" >> $L
pgid=$(cat $D/last_serve.pid)
kill -TERM -$pgid 2> /dev/null
for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
kill -KILL -$pgid 2> /dev/null
export ARM_ROOT=$D/last_arms
(python3 $ARM --label seeded_main --tree $MAIN --card 1 --port 28300 --concurrency 1 --seed 1234 --no-stream >> $D/last_arm_main.log 2>&1; echo "arm seeded_main rc $? $(date +%T)" >> $L) &
(python3 $ARM --label seeded_ours --tree $OURS --card 2 --port 28400 --concurrency 1 --seed 1234 --no-stream >> $D/last_arm_ours.log 2>&1; echo "arm seeded_ours rc $? $(date +%T)" >> $L) &
wait
python3 $ARM --compare seeded_main seeded_ours >> $L 2>&1
echo "queue pr2533 last done $(date +%T)" >> $L
