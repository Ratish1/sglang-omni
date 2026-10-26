#!/usr/bin/env bash
# Base clones on the cascade's (language English) and the SeedTTS client's (auto) requests per server configuration:
# shipped, the x-vector leading-silence mask off (#2357), the whole-text prompt (official non_streaming_mode True),
# whole-text with the mask off, a two-frame minimum (official min_new_tokens 2), and whole-text with the minimum;
# 0.6B and 1.7B, default server settings, through the probe tree (main without its environment toggles). Official
# qwen-tts runs the matching layouts and languages beside it (any arm that fails for memory reruns alone afterwards).
# Then every clip is transcribed (Qwen3-ASR) and every natural stop scored for speaker similarity. Card 1 only.
set -u
D=${D:?the results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
REFPY=${REFPY:?the reference venv python}
CASES=${CASES:?the parity cases}
OLD=${OLD:?the parity results with the official English stream-text arms}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
L=$D/queue.log
mkdir -p $D/served
echo "queue layout start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
cp $OLD/official_06.json $D/official_06_English_stream.json
cp $OLD/official_17.json $D/official_17_English_stream.json
run_official() {
  local size=$1 model=$2 language=$3 layout=$4
  local flags=""
  if [ $layout = whole ]; then
    flags="--whole-text"
  else
    :
  fi
  (cd $D && $REFPY $HERE/parity_reference.py $CASES $D/official_${size}_${language}_${layout}.json --model $model \
    --card $CARD --batch 64 --max-new-tokens $CAP --language $language $flags > $D/official_${size}_${language}_${layout}.log 2>&1)
  local rc=$?
  echo "official $size $language $layout rc $rc $(grep -c 'reached' $D/official_${size}_${language}_${layout}.log) batches $(date +%T)" >> $L
  return $rc
}
OFFICIAL_ARMS="06|Qwen/Qwen3-TTS-12Hz-0.6B-Base|auto|stream 06|Qwen/Qwen3-TTS-12Hz-0.6B-Base|English|whole 06|Qwen/Qwen3-TTS-12Hz-0.6B-Base|auto|whole 17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|auto|stream 17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|English|whole 17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|auto|whole"
(
  for arm in $OFFICIAL_ARMS; do
    IFS='|' read -r size model language layout <<< "$arm"
    run_official $size $model $language $layout
  done
) &
OFFICIAL_PID=$!
start_server() {
  local label=$1 port=$2
  shift 2
  (cd $P && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/serve_$label.pid)
  local waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label server not healthy after $waited s; $(grep -E 'Error|error' $D/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, pgid $pgid, card $(nvidia-smi -i $CARD --query-gpu=memory.used --format=csv,noheader) $(date +%T)" >> $L
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
port=25000
for spec in "06 Qwen/Qwen3-TTS-12Hz-0.6B-Base" "17 Qwen/Qwen3-TTS-12Hz-1.7B-Base"; do
  set -- $spec
  size=$1 model=$2
  for config in "shipped|||icl xvec" \
    "nomask||--tts_engine.factory.leading_silence_mask_frames 0|xvec" \
    "whole|QWEN3_TTS_PROBE_WHOLE_TEXT=1||icl xvec" \
    "whole_nomask|QWEN3_TTS_PROBE_WHOLE_TEXT=1|--tts_engine.factory.leading_silence_mask_frames 0|xvec" \
    "min2|QWEN3_TTS_PROBE_MIN_NEW_TOKENS=2||icl xvec" \
    "whole_min2|QWEN3_TTS_PROBE_WHOLE_TEXT=1 QWEN3_TTS_PROBE_MIN_NEW_TOKENS=2||icl xvec"; do
    IFS='|' read -r name env_vars flags modes <<< "$config"
    label=${size}_$name
    unset QWEN3_TTS_PROBE_WHOLE_TEXT QWEN3_TTS_PROBE_MIN_NEW_TOKENS
    for assignment in $env_vars; do
      export "$assignment"
    done
    echo "$label env [$env_vars] flags [$flags] modes [$modes]" >> $L
    if start_server $label $port --model-path $model --tts_engine.gpu $CARD --vocoder.gpu $CARD $flags; then
      for language in auto English; do
        (cd $P && PYTHONPATH=$P python3 $HERE/layout.py serve $CASES --base-url http://127.0.0.1:$port --model $model \
          --label $label --language $language --out $D/served/${label}_$language --modes $modes \
          --max-new-tokens $CAP >> $L 2>&1)
        echo "served $label $language rc $? $(date +%T)" >> $L
      done
      stop_server $label
    else
      :
    fi
    port=$((port + 200))
  done
done
unset QWEN3_TTS_PROBE_WHOLE_TEXT QWEN3_TTS_PROBE_MIN_NEW_TOKENS
wait $OFFICIAL_PID
for arm in $OFFICIAL_ARMS; do
  IFS='|' read -r size model language layout <<< "$arm"
  if [ ! -s $D/official_${size}_${language}_${layout}.json ]; then
    echo "official $size $language $layout again, alone on the card" >> $L
    run_official $size $model $language $layout
  else
    :
  fi
done
export MASTER_PORT=26850
if start_server asr 26800 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD; then
  (cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $D/served/* --asr-port 26800 >> $L 2>&1)
  echo "transcribed rc $? $(date +%T)" >> $L
  stop_server asr
else
  :
fi
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $D/served/* --sim-card $CARD >> $L 2>&1)
echo "similarity rc $? $(date +%T)" >> $L
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py summary $D/official_*.json --served $D/served/* --max-new-tokens $CAP \
  > $D/summary.txt 2>&1)
echo "summary rc $? $(date +%T)" >> $L
echo "queue_layout_done $(date +%T)" >> $L
