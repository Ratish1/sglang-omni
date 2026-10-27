#!/usr/bin/env bash
# Teacher-forced bisection of 0.6B English runaways: every parity case served by `probe/qwen3-tts-teacher-forced`
# (main plus the two-frame minimum and the mask removal, logging per-step EOS and each frame's codes), streaming PCM,
# cap 300, c16; then official qwen-tts 0.1.1 teacher-forced on the served frames of every runaway and 50 natural stops a
# set, and both compared step by step. The official venv builds beside the serving run. Card 0 only.
set -u
D=${D:?the results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
REVIEW=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=0
CAP=300
PORT=25600
L=$D/queue.log
mkdir -p $D/probe $D/served
echo "queue teacher forced start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
if [ ! -x /data/refenv/bin/python ]; then
  (L=$D/refenv_build.log setsid bash $HERE/refenv_build.sh < /dev/null > /dev/null 2>&1 &)
  echo "refenv build started $(date +%T)" >> $L
else
  echo "refenv_done (present)" >> $D/refenv_build.log
fi
(cd $P && PYTHONPATH=$P python3 $REVIEW/parity.py cases $D >> $L 2>&1)
if [ ! -s $D/cases.jsonl ]; then
  echo "stopped: no cases" >> $L
  echo "queue_teacher_forced_done $(date +%T)" >> $L
  exit 1
else
  echo "cases $(wc -l < $D/cases.jsonl) $(date +%T)" >> $L
fi
(cd $P && QWEN3_TTS_PROBE_DIR=$D/probe SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve.pid; \
  exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --tts_engine.gpu $CARD --vocoder.gpu $CARD" > $D/serve.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve.pid)
WAITED=0
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_teacher_forced_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $REVIEW/layout.py serve $D/cases.jsonl --base-url http://127.0.0.1:$PORT \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --label 06_probe --language English --out $D/served/06_probe \
  --max-new-tokens $CAP >> $L 2>&1)
echo "served rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "stopped $(date +%T), eos lines $(cat $D/probe/eos_*.txt | wc -l), code lines $(cat $D/probe/codes_*.txt | wc -l)" >> $L
until grep -q refenv_done $D/refenv_build.log 2> /dev/null; do
  sleep 20
done
echo "$(grep '^refenv ' $D/refenv_build.log) $(date +%T)" >> $L
/data/refenv/bin/python $HERE/teacher_forced_reference.py $D/served/06_probe $D/probe $D/cases.jsonl \
  $D/official_tf.jsonl --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --card $CARD --natural-per-set 50 >> $L 2>&1
echo "teacher forced rc $? $(date +%T)" >> $L
python3 $HERE/teacher_forced_compare.py $D/probe $D/official_tf.jsonl > $D/compare.txt 2>&1
echo "compare rc $? $(date +%T)" >> $L
echo "queue_teacher_forced_done $(date +%T)" >> $L
