#!/usr/bin/env bash
# Natural utterance lengths per task in English against #3157's default budget and vLLM-Omni's per-task ceilings
# (budget_scan.py): Base 1.7B and 0.6B on SeedTTS EN and its short spans, CustomVoice 1.7B (a native and a non-native
# speaker, plain and slow), VoiceDesign 1.7B (plain and slow). Every server on card 1 only, one at a time.
set -u
D=${D:?the results directory}
M=${M:?the main tree}
T=${T:?the kit tree}
UNITS_LOG=${UNITS_LOG:?the unit queue log}
SCAN=$T/tasks/qwen3-tts-perf/box/review_20261026/budget_scan.py
CARD=1
L=$D/queue.log
mkdir -p $D
until grep -q queue_units_done $UNITS_LOG; do
  sleep 20
done
echo "queue budget start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
for model in Qwen/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-12Hz-0.6B-Base Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign; do
  python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$model'))" >> $D/download.log 2>&1
  echo "download $model rc $? $(date +%T)" >> $L
done
(cd $M && PYTHONPATH=$M python3 $SCAN cases $D/cases >> $L 2>&1)
for cases in base customvoice voicedesign; do
  if [ ! -s $D/cases/$cases.jsonl ]; then
    echo "stopped: no $cases cases" >> $L
    echo "queue_budget_done $(date +%T)" >> $L
    exit 1
  else
    :
  fi
done
run_model() {
  local label=$1 model=$2 port=$3
  shift 3
  (cd $M && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$M setsid bash -c "echo \$\$ > $D/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --model-path $model --host 127.0.0.1 --port $port --tts_engine.gpu $CARD \
    --vocoder.gpu $CARD" > $D/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/serve_$label.pid)
  local waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label server not healthy after $waited s, rc or timeout; $(tail -n 3 $D/serve_$label.log | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, pgid $pgid, card $(nvidia-smi -i $CARD --query-gpu=memory.used --format=csv,noheader) $(date +%T)" >> $L
  for cases in "$@"; do
    (cd $M && PYTHONPATH=$M python3 $SCAN scan $D/cases/$cases.jsonl --base-url http://127.0.0.1:$port --model $model --label $label \
      --out $D/${label}_$cases.jsonl > $D/${label}_$cases.txt 2>&1)
    echo "$label $cases rc $? $(date +%T)" >> $L
    sed "s/^/  /" $D/${label}_$cases.txt >> $L
  done
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do
    kill -0 $pgid 2> /dev/null || break
    sleep 1
  done
  kill -KILL -$pgid 2> /dev/null
  echo "$label stopped $(date +%T)" >> $L
}
run_model base17 Qwen/Qwen3-TTS-12Hz-1.7B-Base 21000 base
run_model base06 Qwen/Qwen3-TTS-12Hz-0.6B-Base 21200 base
run_model customvoice17 Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice 21400 customvoice
run_model voicedesign17 Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign 21600 voicedesign
(cd $M && PYTHONPATH=$M python3 $SCAN summarize $D/base17_base.jsonl $D/base06_base.jsonl \
  $D/customvoice17_customvoice.jsonl $D/voicedesign17_voicedesign.jsonl > $D/summary.txt 2>&1)
echo "queue_budget_done $(date +%T)" >> $L
