#!/usr/bin/env bash
# Block 7's scoring again: the kept outputs run long by selection, and the ASR server's default of four long-audio
# transcriptions in flight answered the rest 503. One Qwen3-ASR server on card 0 that takes 64 at once
# (--audio_chunking.max_concurrent_long_audio_requests), then check, speech and truncated over block 7's scans.
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
SCAN=$T/tasks/qwen3-tts-perf/box/review_20261026/budget_scan.py
MAIN=/workspace/n28_main
W=$D/limit_zh
L=$W/queue.log
RESULTS="$W/customvoice0.jsonl $W/customvoice1.jsonl $W/customvoice2.jsonl $W/voicedesign3.jsonl"
echo "queue block7b start $(date +%T) kit $(git -C $T rev-parse HEAD)" >> $L
(cd $MAIN && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$MAIN setsid bash -c "echo \$\$ > $W/serve_asr2.pid; exec python3 -m sglang_omni.cli serve --host 127.0.0.1 --port 24300 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu 0 --audio_chunking.max_concurrent_long_audio_requests 64" > $W/serve_asr2.log 2>&1 < /dev/null &)
sleep 2
PGID=$(cat $W/serve_asr2.pid)
waited=0
until curl -sf http://127.0.0.1:24300/health > /dev/null; do
  if ! kill -0 $PGID 2> /dev/null || [ $waited -ge 900 ]; then echo "asr2 not healthy" >> $L; exit 1; else sleep 5; waited=$((waited + 5)); fi
done
echo "asr2 healthy after $waited s $(date +%T)" >> $L
(cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN check $RESULTS --asr-port 24300 --concurrency 64 > $W/check.txt 2>&1)
echo "check rc $? $(date +%T)" >> $L
(cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN speech $W/check.jsonl > $W/speech.txt 2>&1)
echo "speech rc $? $(date +%T)" >> $L
(cd $MAIN && PYTHONPATH=$MAIN python3 $SCAN truncated $W/check.jsonl --asr-port 24300 --concurrency 64 > $W/truncated.txt 2>&1)
echo "truncated rc $? $(date +%T)" >> $L
kill -TERM -$PGID 2> /dev/null
for second in $(seq 120); do kill -0 $PGID 2> /dev/null || break; sleep 1; done
kill -KILL -$PGID 2> /dev/null
echo "queue block7b done $(date +%T)" >> $L
