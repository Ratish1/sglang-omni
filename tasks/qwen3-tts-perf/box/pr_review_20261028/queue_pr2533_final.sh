#!/usr/bin/env bash
# #2533 at our final head on the H100, after the first queue (AFTER_LOG holds AFTER_LINE): our serve, qwen3_tts and
# pipeline-state suites on card 0; our speaker-embedding cases called against the PR head (PREFIX: the PR's sources on
# today's main, our final test files copied in with the PR's speaker_embedding_dim spelling, which our head renamed);
# our tree serving 0.6B Base, 1.7B Base and 1.7B CustomVoice on cards 1 to 3 for the exact check from the probe's dumps
# and the contract; then the seeded identity of plain reference requests, main against ours (1.7B Base, c1
# non-streaming, cards 1 and 2).
set -u
D=${D:?the results directory}
OURS=${OURS:?our tree}
MAIN=${MAIN:?the main tree}
PREFIX=${PREFIX:?the PR head tree}
T=${T:?the kit tree}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
HERE=$T/tasks/qwen3-tts-perf/box/pr_review_20261028
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
L=$D/final.log
mkdir -p $D/final_logs
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
echo "queue pr2533 final start $(date +%T) ours $(git -C $OURS rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
for test_file in serve/test_speech_protocol.py qwen3_tts/test_pipeline.py qwen3_tts/test_config.py scheduling/test_pipeline_state.py; do
  sed 's/speaker_embedding_size/speaker_embedding_dim/g' $OURS/tests/unit_test/$test_file > $PREFIX/tests/unit_test/$test_file
done
echo "prefix: $(git -C $PREFIX rev-parse HEAD) with the PR head's $(git -C $PREFIX status --short sglang_omni | tr '\n' ' ')" >> $L
start_server() {
  local label=$1 port=$2
  shift 2
  (cd $OURS && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$OURS setsid bash -c "echo \$\$ > $D/final_logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/final_logs/serve_$label.log 2>&1 < /dev/null &)
}
wait_healthy() {
  local label=$1 port=$2 waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $(cat $D/final_logs/serve_$label.pid) 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label not healthy after $waited s: $(grep -E 'Error|error' $D/final_logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s $(date +%T)" >> $L
}
stop_server() {
  local pgid=$(cat $D/final_logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
  echo "$1 stopped $(date +%T)" >> $L
}
start_server base06 4810 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --tts_engine.gpu 1 --vocoder.gpu 1
start_server base17 4820 --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts_engine.gpu 2 --vocoder.gpu 2
start_server cv17 4830 --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --tts_engine.gpu 3 --vocoder.gpu 3
(cd $OURS && python3 -m pytest tests/unit_test/serve tests/unit_test/qwen3_tts tests/unit_test/scheduling/test_pipeline_state.py \
  -q -p no:cacheprovider > $D/final_units.txt 2>&1)
echo "units ours: $(tail -n 3 $D/final_units.txt | grep -E 'passed|failed|error' | tail -n 1); failing: $(grep -E '^FAILED' $D/final_units.txt | sed 's/ - .*//;s/.*:://' | tr '\n' ' ') $(date +%T)" >> $L
(cd $PREFIX && python3 -m pytest tests/unit_test/serve/test_speech_protocol.py tests/unit_test/qwen3_tts/test_pipeline.py \
  tests/unit_test/qwen3_tts/test_config.py tests/unit_test/scheduling/test_pipeline_state.py -q -p no:cacheprovider \
  -k "speaker_embedding or embedding_the_encoder or round_trips_preserve" > $D/final_prefix.txt 2>&1)
echo "our cases on the PR head: $(grep -E '^FAILED' $D/final_prefix.txt | sed 's/ - .*//;s/.*:://' | tr '\n' ' ') | $(tail -n 1 $D/final_prefix.txt) $(date +%T)" >> $L
wait_healthy base06 4810 || exit 1
wait_healthy base17 4820 || exit 1
wait_healthy cv17 4830 || exit 1
(cd $OURS && PYTHONPATH=$OURS python3 $HERE/speaker_embedding_e2e.py exact --from-dumps --base-url http://127.0.0.1:4810 \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --dump-dir $D/dump06 --out $D/final_exact06.jsonl > $D/final_exact06.txt 2>&1) &
(cd $OURS && PYTHONPATH=$OURS python3 $HERE/speaker_embedding_e2e.py exact --from-dumps --base-url http://127.0.0.1:4820 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --dump-dir $D/dump17 --out $D/final_exact17.jsonl > $D/final_exact17.txt 2>&1) &
wait
echo "exact 0.6B: $(grep -v -i warn $D/final_exact06.txt | tail -n 4 | tr '\n' ';') $(date +%T)" >> $L
echo "exact 1.7B: $(grep -v -i warn $D/final_exact17.txt | tail -n 4 | tr '\n' ';') $(date +%T)" >> $L
REF=$(ls $D/prompt_wavs/*.wav | head -n 1)
for size in 06 17; do
  if [ $size = 06 ]; then port=4810; model=Qwen/Qwen3-TTS-12Hz-0.6B-Base; else port=4820; model=Qwen/Qwen3-TTS-12Hz-1.7B-Base; fi
  (cd $OURS && PYTHONPATH=$OURS python3 $HERE/speaker_embedding_e2e.py contract --base-url http://127.0.0.1:$port --model $model \
    --embedding $(ls $D/dump$size/*.pt | head -n 2 | tail -n 1) --reference $REF --custom-voice-url http://127.0.0.1:4830 \
    --custom-voice-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice > $D/final_contract$size.txt 2>&1)
  echo "contract $size rc $? $(date +%T)" >> $L
done
echo "server tracebacks: 06 $(grep -c Traceback $D/final_logs/serve_base06.log), 17 $(grep -c Traceback $D/final_logs/serve_base17.log), cv $(grep -c Traceback $D/final_logs/serve_cv17.log)" >> $L
for label in base06 base17 cv17; do stop_server $label; done
export ARM_ROOT=$D/final_arms
(python3 $ARM --label seeded_main --tree $MAIN --card 1 --port 27700 --concurrency 1 --seed 1234 --no-stream >> $D/final_logs/arm_main.log 2>&1; echo "arm seeded_main rc $? $(date +%T)" >> $L) &
(python3 $ARM --label seeded_ours --tree $OURS --card 2 --port 27800 --concurrency 1 --seed 1234 --no-stream >> $D/final_logs/arm_ours.log 2>&1; echo "arm seeded_ours rc $? $(date +%T)" >> $L) &
wait
python3 $ARM --compare seeded_main seeded_ours >> $L 2>&1
echo "queue pr2533 final done $(date +%T)" >> $L
