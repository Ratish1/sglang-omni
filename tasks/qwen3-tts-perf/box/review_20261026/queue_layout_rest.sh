#!/usr/bin/env bash
# The rest of queue_layout.sh once its served arms end, reordered to fit the lease: the configuration it points to
# (stream-text prompt, a two-frame minimum, the x-vector leading-silence mask off) served first on 0.6B and 1.7B in both
# languages; then the official arms queue_layout.sh had not reached, two at a time, beside the transcription and
# similarity scoring of every served arm (shipped and the proposed configuration first); then the summary. Card 1 only.
set -u
D=${D:?the layout results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
REFPY=${REFPY:?the reference venv python}
CASES=${CASES:?the parity cases}
RUNNING=${RUNNING:?the pid of the official arm still running}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
L=$D/queue.log
echo "queue layout rest start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
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
export QWEN3_TTS_PROBE_MIN_NEW_TOKENS=2
port=27000
for spec in "06 Qwen/Qwen3-TTS-12Hz-0.6B-Base" "17 Qwen/Qwen3-TTS-12Hz-1.7B-Base"; do
  set -- $spec
  label=${1}_min2_nomask
  echo "$label env [QWEN3_TTS_PROBE_MIN_NEW_TOKENS=2] flags [--tts_engine.factory.leading_silence_mask_frames 0]" >> $L
  if start_server $label $port --model-path $2 --tts_engine.gpu $CARD --vocoder.gpu $CARD \
    --tts_engine.factory.leading_silence_mask_frames 0; then
    for language in English auto; do
      (cd $P && PYTHONPATH=$P python3 $HERE/layout.py serve $CASES --base-url http://127.0.0.1:$port --model $2 \
        --label $label --language $language --out $D/served/${label}_$language --max-new-tokens $CAP >> $L 2>&1)
      echo "served $label $language rc $? $(date +%T)" >> $L
    done
    stop_server $label
  else
    :
  fi
  port=$((port + 200))
done
unset QWEN3_TTS_PROBE_MIN_NEW_TOKENS
while kill -0 $RUNNING 2> /dev/null; do
  sleep 10
done
echo "official 06 English whole ended, $(grep -c 'reached' $D/official_06_English_whole.log) batches, json $(ls $D/official_06_English_whole.json 2> /dev/null) $(date +%T)" >> $L
(
  run_official 06 Qwen/Qwen3-TTS-12Hz-0.6B-Base auto whole
  run_official 17 Qwen/Qwen3-TTS-12Hz-1.7B-Base auto whole
) &
LANE_A=$!
(
  run_official 17 Qwen/Qwen3-TTS-12Hz-1.7B-Base auto stream
  run_official 17 Qwen/Qwen3-TTS-12Hz-1.7B-Base English whole
) &
LANE_B=$!
sleep 300
DIRS=""
for size in 06 17; do
  for arm in shipped min2_nomask; do
    for language in English auto; do
      DIRS="$DIRS $D/served/${size}_${arm}_$language"
    done
  done
done
for size in 06 17; do
  for arm in nomask min2 whole whole_nomask whole_min2; do
    for language in English auto; do
      DIRS="$DIRS $D/served/${size}_${arm}_$language"
    done
  done
done
export MASTER_PORT=27850
if start_server asr 27800 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD; then
  (cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $DIRS --asr-port 27800 --sim-card $CARD >> $L 2>&1)
  echo "transcribed and similarity rc $? $(date +%T)" >> $L
  stop_server asr
else
  :
fi
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py summary $D/official_*.json --served $DIRS --max-new-tokens $CAP \
  > $D/summary.txt 2>&1)
echo "summary scored rc $? $(date +%T)" >> $L
wait $LANE_A $LANE_B
for arm in "06|Qwen/Qwen3-TTS-12Hz-0.6B-Base|English|whole" "06|Qwen/Qwen3-TTS-12Hz-0.6B-Base|auto|whole" \
  "17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|auto|stream" "17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|English|whole" \
  "17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|auto|whole"; do
  IFS='|' read -r size model language layout <<< "$arm"
  if [ ! -s $D/official_${size}_${language}_${layout}.json ]; then
    echo "official $size $language $layout again, alone on the card" >> $L
    run_official $size $model $language $layout
  else
    :
  fi
done
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py summary $D/official_*.json --served $DIRS --max-new-tokens $CAP \
  > $D/summary.txt 2>&1)
echo "summary final rc $? $(date +%T)" >> $L
echo "queue_layout_rest_done $(date +%T)" >> $L
