#!/usr/bin/env bash
# The CI's speaker-loss case on 1.7B Base: after queue_teacher_forced.sh, three parity cases (the two clones of
# reference speaker 10119832 and another low-similarity case) drawn 100 times each on our server as the SeedTTS CI
# client sends them, codes logged by `probe/qwen3-tts-teacher-forced`, then 100 times each on official qwen-tts, both
# scored with the CI's WavLM similarity. Card 0 only.
set -u
D=${D:?the results directory}
TF=${TF:?the teacher-forced directory, for its cases and its done line}
P=${P:?the probe tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
CARD=0
PORT=25800
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
IDS="common_voice_en_10119832-common_voice_en_10119840_icl_sentence common_voice_en_10119832-common_voice_en_10119847_icl_sentence common_voice_en_15735839-common_voice_en_15735837_icl_sentence"
L=$D/queue.log
mkdir -p $D/probe
until grep -q queue_teacher_forced_done $TF/queue.log 2> /dev/null; do
  sleep 20
done
echo "queue speaker loss start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $P && QWEN3_TTS_PROBE_DIR=$D/probe SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve.pid; \
  exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --tts_engine.gpu $CARD --vocoder.gpu $CARD" > $D/serve.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve.pid)
WAITED=0
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_speaker_loss_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/speaker_loss.py serve $TF/cases.jsonl --ids $IDS --base-url http://127.0.0.1:$PORT \
  --model $MODEL --out $D/ours --draws 100 --concurrency 16 >> $L 2>&1)
echo "served rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "stopped $(date +%T)" >> $L
/data/refenv/bin/python $HERE/speaker_loss_reference.py $TF/cases.jsonl --ids $IDS --model $MODEL --out $D/official \
  --draws 100 --batch 16 --card $CARD >> $L 2>&1
echo "official rc $? $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/speaker_loss.py score $D/ours $D/official --card $CARD > $D/summary.txt 2>&1)
echo "scored rc $? $(date +%T)" >> $L
echo "queue_speaker_loss_done $(date +%T)" >> $L
