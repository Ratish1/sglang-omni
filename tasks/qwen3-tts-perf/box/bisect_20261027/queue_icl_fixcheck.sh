#!/usr/bin/env bash
# The CI's collapsing ICL case on one tree, as queue_icl_switches.sh draws it: served at default settings, drawn 300
# times as the SeedTTS CI client sends it (c16), scored with the CI's WavLM similarity; the tree's touched unit suites
# (UNITS, space separated) run beside the draws, which measure quality only. Waits for AFTER_LOG to hold AFTER_LINE.
# Card 0.
set -u
D=${D:?the results directory}
TF=${TF:?the teacher-forced directory, for its cases}
TREE=${TREE:?the server tree}
LABEL=${LABEL:?the arm label}
T=${T:?the kit tree}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
PORT=${PORT:-31600}
UNITS=${UNITS:-}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
IDS=common_voice_en_10119832-common_voice_en_10119840_icl_sentence
L=$D/queue.log
mkdir -p $D
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
echo "queue icl fixcheck $LABEL start $(date +%T) tree $(git -C $TREE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $TREE && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$TREE setsid bash -c "echo \$\$ > $D/serve_$LABEL.pid; \
  exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --tts_engine.gpu 0 --vocoder.gpu 0" > $D/serve_$LABEL.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve_$LABEL.pid)
WAITED=0
HEALTHY=1
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "$LABEL not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve_$LABEL.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    HEALTHY=0
    break
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
if [ $HEALTHY = 1 ]; then
  echo "$LABEL healthy after $WAITED s $(date +%T)" >> $L
  if [ -n "$UNITS" ]; then
    (cd $TREE && timeout 1500 python3 -m pytest -q -p no:cacheprovider $UNITS > $D/units_$LABEL.txt 2>&1; \
      echo "units $LABEL rc $? $(grep -E 'passed|failed' $D/units_$LABEL.txt | tail -n 1) $(date +%T)" >> $L) &
  else
    :
  fi
  (cd $TREE && PYTHONPATH=$TREE python3 $HERE/speaker_loss.py serve $TF/cases.jsonl --ids $IDS \
    --base-url http://127.0.0.1:$PORT --model $MODEL --out $D/$LABEL --draws 300 --concurrency 16 >> $L 2>&1)
  echo "$LABEL served rc $? $(date +%T)" >> $L
  wait
  kill -TERM -$PGID 2> /dev/null
  for second in $(seq 120); do
    kill -0 $PGID 2> /dev/null || break
    sleep 1
  done
  kill -KILL -$PGID 2> /dev/null
  (cd $TREE && PYTHONPATH=$TREE python3 $HERE/speaker_loss.py score $D/$LABEL --card 0 > $D/summary_$LABEL.txt 2>&1)
  echo "scored rc $? $(date +%T)" >> $L
else
  :
fi
echo "queue_icl_fixcheck_${LABEL}_done $(date +%T)" >> $L
