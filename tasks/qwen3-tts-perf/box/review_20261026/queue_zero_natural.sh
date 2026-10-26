#!/usr/bin/env bash
# The EOS forensics run's natural stops that hold an exact-zero run of at least 1 s after the opening, as served and
# ended where that run starts, both transcribed by Qwen3-ASR and printed side by side. Card 1 only.
set -u
D=${D:?the EOS forensics directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
L=$D/zero_natural.log
echo "queue zero natural start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
mkdir -p $D/zero_natural/served
(cd $P && PYTHONPATH=$P python3 $HERE/zero_runs.py cut $D/served/06_probe $D/zero_natural/cut --outcome stop >> $L 2>&1)
echo "cut rc $? $(date +%T)" >> $L
python3 -c "
import json, sys
ids = {json.loads(line)['id'] for line in open('$D/zero_natural/cut/results.jsonl') if line.strip()}
rows = [line for line in open('$D/served/06_probe/results.jsonl') if line.strip() and json.loads(line)['id'] in ids]
open('$D/zero_natural/served/results.jsonl', 'w').writelines(rows)
print(len(rows), 'natural stops with a zero run of at least 1 s after the opening')
" >> $L 2>&1
(cd $P && SGLANG_OMNI_STRICT_PORT=1 MASTER_PORT=28450 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve_asr_natural.pid; exec python3 \
  -m sglang_omni.cli serve --host 127.0.0.1 --port 28400 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD" \
  > $D/serve_asr_natural.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve_asr_natural.pid)
WAITED=0
until curl -sf http://127.0.0.1:28400/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 900 ]; then
    echo "asr server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve_asr_natural.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_zero_natural_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "asr healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $D/zero_natural/served $D/zero_natural/cut --asr-port 28400 >> $L 2>&1)
echo "transcribed rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "asr stopped $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/zero_runs.py report $D/zero_natural/served/scored.jsonl $D/zero_natural/cut \
  > $D/zero_natural_report.txt 2>&1)
echo "report rc $? $(date +%T)" >> $L
echo "queue_zero_natural_done $(date +%T)" >> $L
