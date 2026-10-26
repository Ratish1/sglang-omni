#!/usr/bin/env bash
# Exact-zero runs in the EOS forensics run's audio: zero_runs.py scan over every clip, then the 80 runaways ended at
# their first zero run of at least 1 s, transcribed by Qwen3-ASR and printed against the capped arm. Card 1 only.
set -u
D=${D:?the EOS forensics directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
L=$D/zero.log
echo "queue zero runs start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/zero_runs.py scan $D/served/06_probe > $D/zero_scan.txt 2>&1)
echo "scan rc $? $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/zero_runs.py cut $D/served/06_probe $D/trim/zero >> $L 2>&1)
echo "cut rc $? $(date +%T)" >> $L
(cd $P && SGLANG_OMNI_STRICT_PORT=1 MASTER_PORT=28250 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve_asr_zero.pid; exec python3 \
  -m sglang_omni.cli serve --host 127.0.0.1 --port 28200 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD" \
  > $D/serve_asr_zero.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $D/serve_asr_zero.pid)
WAITED=0
until curl -sf http://127.0.0.1:28200/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $WAITED -ge 900 ]; then
    echo "asr server not healthy after $WAITED s; $(grep -E 'Error|error' $D/serve_asr_zero.log | tail -n 2 | tr '\n' ' ')" >> $L
    kill -TERM -$PGID 2> /dev/null
    echo "queue_zero_runs_done $(date +%T)" >> $L
    exit 1
  else
    sleep 5
    WAITED=$((WAITED + 5))
  fi
done
echo "asr healthy after $WAITED s, pgid $PGID $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $D/trim/zero --asr-port 28200 >> $L 2>&1)
echo "transcribed rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do
  kill -0 $PGID 2> /dev/null || break
  sleep 1
done
kill -KILL -$PGID 2> /dev/null
echo "asr stopped $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/zero_runs.py report $D/trim $D/trim/zero > $D/zero_report.txt 2>&1)
echo "report rc $? $(date +%T)" >> $L
echo "queue_zero_runs_done $(date +%T)" >> $L
