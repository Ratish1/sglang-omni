#!/usr/bin/env bash
# Block 5 of 10-28: what DECISIONS 112 asks before CustomVoice and VoiceDesign take the checkpoint's limit: their
# runaways on main with an explicit 2048 (budget_scan.py scan: seeded, non-streaming, c32) on every CustomVoice speaker
# over every SeedTTS EN and ZH sentence, the slow instruction on ZH, and VoiceDesign plain and slow on ZH. Three
# CustomVoice servers (cards 0 to 2, a third of the cases each) and one VoiceDesign server (card 3). Waits for block 4.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
SCAN=$T/tasks/qwen3-tts-perf/box/review_20261026/budget_scan.py
MAIN=/workspace/n28_main
CV=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
VD=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
W=$D/wide
L=$W/queue.log
mkdir -p $W
until grep -q "queue block4 done" $D/bar/queue.log 2> /dev/null; do sleep 30; done
echo "queue block5 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
(cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN cases-wide $W/cases --custom-voice-model $CV >> $L 2>&1)
split -n l/3 -d --additional-suffix=.jsonl $W/cases/customvoice_wide.jsonl $W/cases/customvoice_part
echo "cases $(wc -l < $W/cases/customvoice_wide.jsonl) customvoice in $(ls $W/cases/customvoice_part*.jsonl | wc -l) parts, $(wc -l < $W/cases/voicedesign_wide.jsonl) voicedesign $(date +%T)" >> $L

scan_one() {
  local label=$1 card=$2 port=$3 model=$4 cases=$5
  (cd $MAIN && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$MAIN setsid bash -c "echo \$\$ > $W/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port --model-path $model --tts_engine.gpu $card --vocoder.gpu $card" > $W/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then echo "$label not healthy" >> $L; return 1; else sleep 5; waited=$((waited + 5)); fi
  done
  echo "$label healthy after $waited s $(date +%T)" >> $L
  (cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN scan $cases --base-url http://127.0.0.1:$port --model $model --label ${label%[0-9]} \
    --out $W/$label.jsonl --concurrency 32 > $W/$label.summary.txt 2>&1)
  echo "$label scanned rc $? $(date +%T)" >> $L
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}
pids=()
scan_one customvoice0 0 6000 $CV $W/cases/customvoice_part00.jsonl & pids+=($!)
scan_one customvoice1 1 6500 $CV $W/cases/customvoice_part01.jsonl & pids+=($!)
scan_one customvoice2 2 7000 $CV $W/cases/customvoice_part02.jsonl & pids+=($!)
scan_one voicedesign3 3 7500 $VD $W/cases/voicedesign_wide.jsonl & pids+=($!)
wait ${pids[@]}
(cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN summarize $W/customvoice0.jsonl $W/customvoice1.jsonl $W/customvoice2.jsonl $W/voicedesign3.jsonl > $W/summary.txt 2>&1)
echo "summary rc $? $(date +%T)" >> $L
echo "queue block5 done $(date +%T)" >> $L
