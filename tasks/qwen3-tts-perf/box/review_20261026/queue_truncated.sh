#!/usr/bin/env bash
# Every checked output that speaks the target, cut where #3157's budget and vLLM-Omni's ceiling would stop it and
# transcribed again on Qwen3-ASR (budget_scan.py truncated): the cuts that drop target words. Card 1 only.
set -u
D=${D:?the validity results directory}
M=${M:?the main tree}
T=${T:?the kit tree}
SCAN=$T/tasks/qwen3-tts-perf/box/review_20261026/budget_scan.py
CARD=1
PORT=21800
L=$D/queue.log
echo "queue truncated start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
export MASTER_PORT=21850
(cd $M && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$M setsid bash -c "echo \$\$ > $D/serve_asr2.pid; exec python3 \
  -m sglang_omni.cli serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $PORT --asr.gpu $CARD" \
  > $D/serve_asr2.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve_asr2.pid)
WAITED=0
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "asr server not healthy after $WAITED s" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_truncated_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
(cd $M && PYTHONPATH=$M python3 $SCAN truncated $D/check.jsonl --asr-port $PORT > $D/truncated.txt 2>&1)
echo "truncated rc $? $(date +%T)" >> $L
grep -E "^set |^base|^customvoice|^voicedesign|Error|Traceback" $D/truncated.txt >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "queue_truncated_done $(date +%T)" >> $L
