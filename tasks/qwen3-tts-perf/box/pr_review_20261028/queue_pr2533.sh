#!/usr/bin/env bash
# #2533 on the H100: the probe build (the PR with upstream main, plus a dump of each computed reference embedding)
# serves 0.6B Base on card 0 and 1.7B Base on card 1, the PR's CustomVoice 1.7B serves on card 2; card 3 runs the
# serve and qwen3_tts suites on main and the PR. Then the exact runs (both sizes at once) and the contract runs.
set -u
D=${D:?the results directory}
PROBE=${PROBE:?the probe tree}
PR=${PR:?the PR tree}
MAIN=${MAIN:?the main tree}
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/pr_review_20261028
L=$D/queue.log
mkdir -p $D/logs $D/dump06 $D/dump17
echo "queue pr2533 start $(date +%T) probe $(git -C $PROBE rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')pr $(git -C $PR rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
start_server() {
  local label=$1 tree=$2 port=$3 dump=$4
  shift 4
  (cd $tree && QWEN3_TTS_PROBE_SPEAKER_DIR=$dump SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$tree setsid bash -c "echo \$\$ > $D/logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/logs/serve_$label.log 2>&1 < /dev/null &)
}
wait_healthy() {
  local label=$1 port=$2 waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $(cat $D/logs/serve_$label.pid) 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label not healthy after $waited s: $(grep -E 'Error|error' $D/logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s $(date +%T)" >> $L
}
stop_server() {
  local pgid=$(cat $D/logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
  echo "$1 stopped $(date +%T)" >> $L
}
start_server base06 $PROBE 4610 $D/dump06 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --tts_engine.gpu 0 --vocoder.gpu 0
start_server base17 $PROBE 4620 $D/dump17 --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts_engine.gpu 1 --vocoder.gpu 1
start_server cv17 $PR 4630 "" --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --tts_engine.gpu 2 --vocoder.gpu 2
(cd $PR && PYTHONPATH=$PR python3 $T/tasks/qwen3-tts-perf/box/bisect_20261027/stage_seedtts_wavs.py $D/wavs > $D/logs/stage.log 2>&1)
echo "staged $(ls $D/wavs | wc -l) references $(date +%T)" >> $L
for tree in main pr; do
  if [ $tree = main ]; then dir=$MAIN; else dir=$PR; fi
  (cd $dir && python3 -m pytest tests/unit_test/serve tests/unit_test/qwen3_tts -q -p no:cacheprovider > $D/units_$tree.txt 2>&1)
  echo "units $tree: $(tail -n 3 $D/units_$tree.txt | grep -E 'passed|failed|error' | tail -n 1) $(date +%T)" >> $L
done
wait_healthy base06 4610 || exit 1
wait_healthy base17 4620 || exit 1
wait_healthy cv17 4630 || exit 1
(cd $PR && PYTHONPATH=$PR python3 $HERE/speaker_embedding_e2e.py exact --base-url http://127.0.0.1:4610 \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --dump-dir $D/dump06 --out $D/exact06.jsonl > $D/exact06.txt 2>&1) &
(cd $PR && PYTHONPATH=$PR python3 $HERE/speaker_embedding_e2e.py exact --base-url http://127.0.0.1:4620 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --dump-dir $D/dump17 --out $D/exact17.jsonl > $D/exact17.txt 2>&1) &
wait
echo "exact 0.6B: $(grep -v -i warn $D/exact06.txt | tail -n 4 | tr '\n' ';') $(date +%T)" >> $L
echo "exact 1.7B: $(grep -v -i warn $D/exact17.txt | tail -n 4 | tr '\n' ';') $(date +%T)" >> $L
REF=$(ls $D/wavs/*.wav | head -n 1)
for size in 06 17; do
  if [ $size = 06 ]; then port=4610; model=Qwen/Qwen3-TTS-12Hz-0.6B-Base; else port=4620; model=Qwen/Qwen3-TTS-12Hz-1.7B-Base; fi
  (cd $PR && PYTHONPATH=$PR python3 $HERE/speaker_embedding_e2e.py contract --base-url http://127.0.0.1:$port --model $model \
    --embedding $(ls $D/dump$size/*.pt | head -n 1) --reference $REF --custom-voice-url http://127.0.0.1:4630 \
    --custom-voice-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice > $D/contract$size.txt 2>&1)
  echo "contract $size rc $? $(date +%T)" >> $L
done
echo "server errors: 06 $(grep -c -E 'Traceback|device-side assert|CUDA error' $D/logs/serve_base06.log), 17 $(grep -c -E 'Traceback|device-side assert|CUDA error' $D/logs/serve_base17.log)" >> $L
for label in base06 base17 cv17; do stop_server $label; done
echo "queue pr2533 done $(date +%T)" >> $L
