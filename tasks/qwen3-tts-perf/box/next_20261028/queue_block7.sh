#!/usr/bin/env bash
# Block 7 of 10-28: what #3157's text budget cuts on CustomVoice and VoiceDesign Chinese, read from the audio. Every
# CustomVoice ZH case (11 sets, 22,220) on three main servers (cards 0 to 2, a third each) and every VoiceDesign ZH
# case (plain and slow, 4,040) on card 3, as block 5 sent them (seeded, non-streaming, c32, explicit 2048), keeping
# the audio of every natural stop past the budget (--save-over). Then one Qwen3-ASR server on card 0: check sorts each
# kept output into the target spoken, a loop or unclear (Chinese per character), speech says whether speech goes on
# past the cut, and truncated transcribes each spoken output cut at the budget and counts the target characters lost.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
SCAN=$T/tasks/qwen3-tts-perf/box/review_20261026/budget_scan.py
MAIN=/workspace/n28_main
CV=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
VD=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
W=$D/limit_zh
L=$W/queue.log
mkdir -p $W/cases
echo "queue block7 start $(date +%T) main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
grep -F '"set": "customvoice zh ' $D/wide/cases/customvoice_wide.jsonl > $W/cases/customvoice_zh.jsonl
split -n l/3 -d --additional-suffix=.jsonl $W/cases/customvoice_zh.jsonl $W/cases/customvoice_zh_part
echo "cases $(wc -l < $W/cases/customvoice_zh.jsonl) customvoice zh, $(wc -l < $D/wide/cases/voicedesign_wide.jsonl) voicedesign zh $(date +%T)" >> $L

start_server() {
  local label=$1 port=$2
  shift 2
  (cd $MAIN && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$MAIN setsid bash -c "echo \$\$ > $W/serve_$label.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $W/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $W/serve_$label.pid) waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then echo "$label not healthy" >> $L; return 1; else sleep 5; waited=$((waited + 5)); fi
  done
  echo "$label healthy after $waited s $(date +%T)" >> $L
}

stop_server() {
  local pgid=$(cat $W/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
}

scan_one() {
  local label=$1 card=$2 port=$3 model=$4 cases=$5
  start_server $label $port --model-path $model --tts_engine.gpu $card --vocoder.gpu $card || return 1
  (cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN scan $cases --base-url http://127.0.0.1:$port --model $model --label ${label%[0-9]} \
    --out $W/$label.jsonl --concurrency 32 --save-over > $W/$label.summary.txt 2>&1)
  echo "$label scanned rc $? errors $(grep -c '"error"' $W/$label.jsonl) kept $(grep -c '"audio"' $W/$label.jsonl) $(date +%T)" >> $L
  stop_server $label
}
pids=()
scan_one customvoice0 0 6200 $CV $W/cases/customvoice_zh_part00.jsonl & pids+=($!)
scan_one customvoice1 1 6700 $CV $W/cases/customvoice_zh_part01.jsonl & pids+=($!)
scan_one customvoice2 2 7200 $CV $W/cases/customvoice_zh_part02.jsonl & pids+=($!)
scan_one voicedesign3 3 7700 $VD $D/wide/cases/voicedesign_wide.jsonl & pids+=($!)
wait ${pids[@]}
RESULTS="$W/customvoice0.jsonl $W/customvoice1.jsonl $W/customvoice2.jsonl $W/voicedesign3.jsonl"
(cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN summarize $RESULTS > $W/summary.txt 2>&1)

if start_server asr 23900 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0; then
  (cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN check $RESULTS --asr-port 23900 --concurrency 64 > $W/check.txt 2>&1)
  echo "check rc $? $(date +%T)" >> $L
  (cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN speech $W/check.jsonl > $W/speech.txt 2>&1)
  echo "speech rc $? $(date +%T)" >> $L
  (cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN truncated $W/check.jsonl --asr-port 23900 --concurrency 64 > $W/truncated.txt 2>&1)
  echo "truncated rc $? $(date +%T)" >> $L
  stop_server asr
else
  echo "asr not started, nothing scored" >> $L
fi
echo "queue block7 done $(date +%T)" >> $L
