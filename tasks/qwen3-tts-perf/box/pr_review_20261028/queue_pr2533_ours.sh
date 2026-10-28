#!/usr/bin/env bash
# #2533 with our changes on the H100. At once: the SeedTTS EN reference clips staged, and a venv holding only qwen-tts
# 0.1.1 and its pinned dependencies. After the #3284 queue frees the cards (AFTER_LOG holds AFTER_LINE): our serve and
# qwen3_tts suites on card 0 and our new cases called against the PR head (PREFIX, our test files over its code), while
# our tree serves 0.6B Base on card 1, 1.7B Base on card 2 and 1.7B CustomVoice on card 3; then the embedding request
# against the reference request byte for byte (embeddings the PR's probe build dumped), the contract on both sizes, the
# servers stopped; then qwen-tts's embedding against the server's (cards 1 and 2) and the seeded identity of plain
# reference requests, main against ours (1.7B Base, c1 non-streaming, cards 1 and 2).
set -u
D=${D:?the results directory}
OURS=${OURS:?our tree}
MAIN=${MAIN:?the main tree}
PREFIX=${PREFIX:?the PR head tree carrying our test files}
T=${T:?the kit tree}
AFTER_LOG=${AFTER_LOG:?the log to wait on}
AFTER_LINE=${AFTER_LINE:?the line to wait for}
HERE=$T/tasks/qwen3-tts-perf/box/pr_review_20261028
ARM=$T/tasks/qwen3-tts-perf/box/inline_20261026/serve_arm.py
VENV=$D/qwen_tts_venv
L=$D/ours.log
mkdir -p $D/ours_logs
echo "queue pr2533 ours start $(date +%T) ours $(git -C $OURS rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')main $(git -C $MAIN rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')prefix $(git -C $PREFIX rev-parse HEAD) kit $(git -C $T rev-parse HEAD)" >> $L
(python3 -m venv $VENV && $VENV/bin/pip install -q qwen-tts==0.1.1 > $D/ours_logs/venv.log 2>&1; echo "venv rc $? $($VENV/bin/pip list 2> /dev/null | grep -iE '^(qwen-tts|transformers|torch) ' | tr -s ' ' | tr '\n' ';') $(date +%T)" >> $L) &
(cd $OURS && PYTHONPATH=$OURS python3 $HERE/qwen_tts_embedding_compare.py stage --wav-dir $D/prompt_wavs > $D/ours_logs/stage.log 2>&1)
echo "staged: $(tail -n 1 $D/ours_logs/stage.log) $(date +%T)" >> $L
until grep -q "$AFTER_LINE" $AFTER_LOG 2> /dev/null; do
  sleep 20
done
echo "cards free $(date +%T)" >> $L
start_server() {
  local label=$1 port=$2
  shift 2
  (cd $OURS && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$OURS setsid bash -c "echo \$\$ > $D/ours_logs/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/ours_logs/serve_$label.log 2>&1 < /dev/null &)
}
wait_healthy() {
  local label=$1 port=$2 waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $(cat $D/ours_logs/serve_$label.pid) 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label not healthy after $waited s: $(grep -E 'Error|error' $D/ours_logs/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s $(date +%T)" >> $L
}
stop_server() {
  local pgid=$(cat $D/ours_logs/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do kill -0 $pgid 2> /dev/null || break; sleep 1; done
  kill -KILL -$pgid 2> /dev/null
  echo "$1 stopped $(date +%T)" >> $L
}
start_server base06 4710 --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base --tts_engine.gpu 1 --vocoder.gpu 1
start_server base17 4720 --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts_engine.gpu 2 --vocoder.gpu 2
start_server cv17 4730 --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --tts_engine.gpu 3 --vocoder.gpu 3
(cd $OURS && python3 -m pytest tests/unit_test/serve tests/unit_test/qwen3_tts -q -p no:cacheprovider > $D/ours_units.txt 2>&1)
echo "units ours: $(tail -n 3 $D/ours_units.txt | grep -E 'passed|failed|error' | tail -n 1) $(date +%T)" >> $L
(cd $PREFIX && python3 -m pytest tests/unit_test/serve/test_speech_protocol.py tests/unit_test/qwen3_tts/test_pipeline.py \
  tests/unit_test/qwen3_tts/test_config.py -q -p no:cacheprovider -k "speaker_embedding or embedding_the_encoder" > $D/prefix_units.txt 2>&1)
echo "new cases on the PR head: $(grep -E '^FAILED' $D/prefix_units.txt | sed 's/ - .*//' | tr '\n' ' ') | $(tail -n 1 $D/prefix_units.txt) $(date +%T)" >> $L
wait_healthy base06 4710 || exit 1
wait_healthy base17 4720 || exit 1
wait_healthy cv17 4730 || exit 1
(cd $OURS && PYTHONPATH=$OURS python3 $HERE/speaker_embedding_e2e.py exact --from-dumps --base-url http://127.0.0.1:4710 \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --dump-dir $D/dump06 --out $D/ours_exact06.jsonl > $D/ours_exact06.txt 2>&1) &
(cd $OURS && PYTHONPATH=$OURS python3 $HERE/speaker_embedding_e2e.py exact --from-dumps --base-url http://127.0.0.1:4720 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --dump-dir $D/dump17 --out $D/ours_exact17.jsonl > $D/ours_exact17.txt 2>&1) &
wait
echo "exact 0.6B: $(grep -v -i warn $D/ours_exact06.txt | tail -n 4 | tr '\n' ';') $(date +%T)" >> $L
echo "exact 1.7B: $(grep -v -i warn $D/ours_exact17.txt | tail -n 4 | tr '\n' ';') $(date +%T)" >> $L
REF=$(ls $D/prompt_wavs/*.wav | head -n 1)
for size in 06 17; do
  if [ $size = 06 ]; then port=4710; model=Qwen/Qwen3-TTS-12Hz-0.6B-Base; else port=4720; model=Qwen/Qwen3-TTS-12Hz-1.7B-Base; fi
  (cd $OURS && PYTHONPATH=$OURS python3 $HERE/speaker_embedding_e2e.py contract --base-url http://127.0.0.1:$port --model $model \
    --embedding $(ls $D/dump$size/*.pt | head -n 2 | tail -n 1) --reference $REF --custom-voice-url http://127.0.0.1:4730 \
    --custom-voice-model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice > $D/ours_contract$size.txt 2>&1)
  echo "contract $size rc $? $(date +%T)" >> $L
done
echo "server errors: 06 $(grep -c -E 'Traceback|device-side assert|CUDA error' $D/ours_logs/serve_base06.log), 17 $(grep -c -E 'Traceback|device-side assert|CUDA error' $D/ours_logs/serve_base17.log)" >> $L
for label in base06 base17 cv17; do stop_server $label; done
wait
(cd $D && $VENV/bin/python $HERE/qwen_tts_embedding_compare.py compare --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --dump-dir $D/dump06 --wav-dir $D/prompt_wavs --device cuda:1 > $D/qwen_tts_compare06.txt 2>&1; echo "qwen-tts 0.6B rc $?: $(tail -n 2 $D/qwen_tts_compare06.txt | tr '\n' ' ') $(date +%T)" >> $L) &
(cd $D && $VENV/bin/python $HERE/qwen_tts_embedding_compare.py compare --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --dump-dir $D/dump17 --wav-dir $D/prompt_wavs --device cuda:2 > $D/qwen_tts_compare17.txt 2>&1; echo "qwen-tts 1.7B rc $?: $(tail -n 2 $D/qwen_tts_compare17.txt | tr '\n' ' ') $(date +%T)" >> $L) &
wait
export ARM_ROOT=$D/arms
(python3 $ARM --label seeded_main --tree $MAIN --card 1 --port 27500 --concurrency 1 --seed 1234 --no-stream >> $D/ours_logs/arm_main.log 2>&1; echo "arm seeded_main rc $? $(date +%T)" >> $L) &
(python3 $ARM --label seeded_ours --tree $OURS --card 2 --port 27600 --concurrency 1 --seed 1234 --no-stream >> $D/ours_logs/arm_ours.log 2>&1; echo "arm seeded_ours rc $? $(date +%T)" >> $L) &
wait
python3 $ARM --compare seeded_main seeded_ours >> $L 2>&1
echo "queue pr2533 ours done $(date +%T)" >> $L
