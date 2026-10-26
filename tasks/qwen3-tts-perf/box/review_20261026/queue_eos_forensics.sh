#!/usr/bin/env bash
# EOS forensics: `probe/qwen3-tts-eos-forensics` (the two-frame minimum and the mask removal, plus a per-step EOS log)
# on 0.6B, every parity case in English as the cascade posts it, cap 300, c16; then eos_forensics.py joins the log to
# the outcomes. Card 1 only.
set -u
D=${D:?the results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
CASES=${CASES:?the parity cases}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
L=$D/queue.log
mkdir -p $D/probe $D/served
echo "queue eos forensics start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $P && QWEN3_TTS_PROBE_EOS_DIR=$D/probe SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve.pid; \
  exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port 25500 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --tts_engine.gpu $CARD --vocoder.gpu $CARD" > $D/serve.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve.pid)
WAITED=0
until curl -sf http://127.0.0.1:25500/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_eos_forensics_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py serve $CASES --base-url http://127.0.0.1:25500 \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --label 06_probe --language English --out $D/served/06_probe \
  --max-new-tokens 300 >> $L 2>&1)
echo "served rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "stopped $(date +%T), probe lines $(cat $D/probe/eos_*.txt | wc -l)" >> $L
python3 $HERE/eos_forensics.py $D/served/06_probe $D/probe > $D/forensics.txt 2>&1
echo "forensics rc $? $(date +%T)" >> $L
echo "queue_eos_forensics_done $(date +%T)" >> $L
