#!/usr/bin/env bash
# The EOS forensics run's 80 runaways ended at their peak-EOS step against the same runaways kept to the cap: eos_trim.py
# cut writes both, Qwen3-ASR transcribes both, eos_trim.py report prints them case by case. Card 1 only.
set -u
D=${D:?the EOS forensics directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
L=$D/trim.log
echo "queue eos trim start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/eos_trim.py cut $D/served/06_probe $D/probe $D/trim > $D/trim_cut.txt 2>&1)
echo "cut rc $? $(date +%T)" >> $L
(cd $P && SGLANG_OMNI_STRICT_PORT=1 MASTER_PORT=28050 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve_asr.pid; exec python3 \
  -m sglang_omni.cli serve --host 127.0.0.1 --port 28000 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD" \
  > $D/serve_asr.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve_asr.pid)
WAITED=0
until curl -sf http://127.0.0.1:28000/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 900 ]; then
    echo "asr server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve_asr.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_eos_trim_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "asr healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $D/trim/capped $D/trim/peak --asr-port 28000 >> $L 2>&1)
echo "transcribed rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "asr stopped $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/eos_trim.py report $D/trim > $D/trim_report.txt 2>&1)
echo "report rc $? $(date +%T)" >> $L
echo "queue_eos_trim_done $(date +%T)" >> $L
