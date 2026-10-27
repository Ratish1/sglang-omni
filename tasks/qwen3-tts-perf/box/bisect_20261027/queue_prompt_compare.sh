#!/usr/bin/env bash
# Our talker prompt against official's, 0.6B English: after queue_speaker_loss.sh, every runaway case of the
# teacher-forced run plus 20 natural stops a set served again by `probe/qwen3-tts-teacher-forced` with
# QWEN3_TTS_PROBE_PROMPT_DIR set, then prompt_compare_reference.py beside official qwen-tts. Card 0 only.
set -u
D=${D:?the results directory}
TF=${TF:?the teacher-forced directory}
SPK=${SPK:?the speaker-loss directory, for its done line}
P=${P:?the probe tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/bisect_20261027
REVIEW=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=0
PORT=26000
MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
L=$D/queue.log
mkdir -p $D/prompts $D/served
until grep -q queue_speaker_loss_done $SPK/queue.log 2> /dev/null; do
  sleep 20
done
echo "queue prompt compare start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
python3 - $TF/served/06_probe/results.jsonl $TF/cases.jsonl $D/cases.jsonl >> $L 2>&1 <<'EOF'
import json, sys
from collections import defaultdict
served = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
chosen = {record["id"] for record in served if record.get("finish_reason") == "length"}
natural = defaultdict(list)
for record in sorted(served, key=lambda record: record["id"]):
    if record.get("finish_reason") == "stop":
        natural[record["set"]].append(record["id"])
for ids in natural.values():
    chosen.update(ids[:: max(1, len(ids) // 20)][:20])
cases = [line for line in open(sys.argv[2]) if line.strip() and json.loads(line)["id"] in chosen]
open(sys.argv[3], "w").writelines(cases)
print(f"cases {len(cases)}")
EOF
(cd $P && QWEN3_TTS_PROBE_PROMPT_DIR=$D/prompts SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve.pid; \
  exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $PORT --model-path $MODEL \
  --tts_engine.gpu $CARD --vocoder.gpu $CARD" > $D/serve.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve.pid)
WAITED=0
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 1800 ]; then
    echo "server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_prompt_compare_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $REVIEW/layout.py serve $D/cases.jsonl --base-url http://127.0.0.1:$PORT \
  --model $MODEL --label 06_prompt --language English --out $D/served/06_prompt --max-new-tokens 300 >> $L 2>&1)
echo "served rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "stopped $(date +%T), prompts $(ls $D/prompts | wc -l)" >> $L
/data/refenv/bin/python $HERE/prompt_compare_reference.py $D/cases.jsonl $D/served/06_prompt $D/prompts \
  --model $MODEL --card $CARD > $D/compare.txt 2>&1
echo "compare rc $? $(date +%T)" >> $L
echo "queue_prompt_compare_done $(date +%T)" >> $L
