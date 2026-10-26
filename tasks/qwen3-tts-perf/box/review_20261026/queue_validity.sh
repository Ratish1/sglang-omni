#!/usr/bin/env bash
# The English scan again with every request's task type sent (VoiceDesign failed without it) and the audio of each
# natural stop past #3157's budget kept, then Qwen3-ASR on the same card sorts those outputs into the target spoken,
# loops and unclear (budget_scan.py check). Every server on card 1 only, one at a time.
set -u
D=${D:?the results directory}
M=${M:?the main tree}
T=${T:?the kit tree}
SCAN=$T/tasks/qwen3-tts-perf/box/review_20261026/budget_scan.py
CARD=1
L=$D/queue.log
mkdir -p $D
echo "queue validity start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
(cd $M && PYTHONPATH=$M python3 $SCAN cases $D/cases >> $L 2>&1)
for cases in base customvoice voicedesign; do
  if [ ! -s $D/cases/$cases.jsonl ]; then
    echo "stopped: no $cases cases" >> $L
    echo "queue_validity_done $(date +%T)" >> $L
    exit 1
  else
    :
  fi
done
start_server() {
  local label=$1 port=$2
  shift 2
  (cd $M && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$M setsid bash -c "echo \$\$ > $D/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/serve_$label.pid)
  local waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label server not healthy after $waited s; $(tail -n 3 $D/serve_$label.log | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, pgid $pgid $(date +%T)" >> $L
}
stop_server() {
  local pgid=$(cat $D/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do
    kill -0 $pgid 2> /dev/null || break
    sleep 1
  done
  kill -KILL -$pgid 2> /dev/null
  echo "$1 stopped $(date +%T)" >> $L
}
for spec in "base17 Qwen/Qwen3-TTS-12Hz-1.7B-Base 21000 base" "base06 Qwen/Qwen3-TTS-12Hz-0.6B-Base 21200 base" \
  "customvoice17 Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice 21400 customvoice" \
  "voicedesign17 Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign 21600 voicedesign"; do
  set -- $spec
  if start_server $1 $3 --model-path $2 --tts_engine.gpu $CARD --vocoder.gpu $CARD; then
    (cd $M && PYTHONPATH=$M python3 $SCAN scan $D/cases/$4.jsonl --base-url http://127.0.0.1:$3 --model $2 --label $1 \
      --out $D/${1}_$4.jsonl --save-over > $D/${1}_$4.txt 2>&1)
    echo "$1 $4 rc $? $(date +%T)" >> $L
    grep -E "^$1 |^  error" $D/${1}_$4.txt >> $L
    stop_server $1
  else
    :
  fi
done
# note (ratish): the ASR stage takes its rendezvous port from MASTER_PORT, kept outside the 29500 block.
export MASTER_PORT=21850
if start_server asr 21800 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD; then
  (cd $M && PYTHONPATH=$M python3 $SCAN check $D/base17_base.jsonl $D/base06_base.jsonl $D/customvoice17_customvoice.jsonl \
    $D/voicedesign17_voicedesign.jsonl --asr-port 21800 > $D/check.txt 2>&1)
  echo "check rc $? $(date +%T)" >> $L
  cat $D/check.txt | grep -vE "Warning|warn" >> $L
  stop_server asr
else
  :
fi
(cd $M && PYTHONPATH=$M python3 $SCAN summarize $D/base17_base.jsonl $D/base06_base.jsonl \
  $D/customvoice17_customvoice.jsonl $D/voicedesign17_voicedesign.jsonl > $D/summary.txt 2>&1)
echo "queue_validity_done $(date +%T)" >> $L
